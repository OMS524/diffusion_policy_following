#!/usr/bin/env python3
#
# Copyright 2025 AIRO LABS., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DINOv2-anchored Lucas-Kanade keypoint tracker for cam_top.

Subscribes to:
  /cam_top/image_compressed  sensor_msgs/CompressedImage
  /cam_top/depth_raw         sensor_msgs/Image (16UC1)
  /cam_top/camera_info       sensor_msgs/CameraInfo

Publishes:
  /keypoints/distances       std_msgs/Float32MultiArray  ([d_right_obj, d_left_obj])

Workflow:
  1. First color frame opens an OpenCV window. The user clicks three keypoints in
     order:  kp0=right gripper, kp1=object, kp2=left gripper.
  2. DINOv2 reference embeddings are stored for each click.
  3. Each subsequent frame is tracked with Lucas-Kanade. If a track fails
     (status=0), the node searches a 100x100 ROI around the last known position
     and snaps to the location with highest cosine similarity to the reference.
  4. Pixel positions are back-projected to camera frame using depth + intrinsics.
  5. The Euclidean distances right<->object and left<->object are published at
     ~10 Hz with the latest tracked positions (forward-fill on stale frames).

Keys (focus the display window):
  r  re-initialise — clear keypoints and reopen click capture.
  q  shut down.
"""

import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Float32MultiArray

from dp_camera._dino import DinoFeatureExtractor


WINDOW_NAME = "dp_camera"
LABELS = ["right_gripper", "object", "left_gripper"]
DRAW_COLOURS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]


@dataclass
class TrackState:
    pts: np.ndarray              # shape (N, 1, 2), float32
    refs: list                   # list of torch.Tensor (DINO embeddings)
    last_xyz: np.ndarray         # shape (N, 3), float32 (camera frame, metres)
    valid: np.ndarray            # shape (N,) bool — last back-projection succeeded


class KeypointNode(Node):

    def __init__(self):
        super().__init__("keypoint_node")

        self.declare_parameter("model_device", "cuda")
        self.declare_parameter("dino_patch_size", 56)
        self.declare_parameter("n_keypoints", 3)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("recovery_roi", 100)
        self.declare_parameter("recovery_stride", 8)
        self.declare_parameter("recovery_threshold", 0.5)
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("color_topic", "/cam_top/image_compressed")
        self.declare_parameter("depth_topic", "/cam_top/depth_raw")
        self.declare_parameter("camera_info_topic", "/cam_top/camera_info")
        self.declare_parameter("distances_topic", "/keypoints/distances")

        self.n_keypoints = int(self.get_parameter("n_keypoints").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.recovery_roi = int(self.get_parameter("recovery_roi").value)
        self.recovery_stride = int(self.get_parameter("recovery_stride").value)
        self.recovery_threshold = float(self.get_parameter("recovery_threshold").value)
        publish_rate = float(self.get_parameter("publish_rate").value)

        self.bridge = CvBridge()
        self.dino = DinoFeatureExtractor(
            device=str(self.get_parameter("model_device").value),
            patch_size=int(self.get_parameter("dino_patch_size").value),
        )

        self._latest_color: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._fx = self._fy = self._ppx = self._ppy = None
        self._track: Optional[TrackState] = None
        self._click_buffer: list = []
        self._awaiting_clicks = False
        self._lock = threading.Lock()
        self._shutdown = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(
            CompressedImage,
            self.get_parameter("color_topic").value,
            self._color_cb,
            sensor_qos,
        )
        self.create_subscription(
            Image,
            self.get_parameter("depth_topic").value,
            self._depth_cb,
            sensor_qos,
        )
        self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._info_cb,
            sensor_qos,
        )
        self.pub_distances = self.create_publisher(
            Float32MultiArray,
            self.get_parameter("distances_topic").value,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST),
        )

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.create_timer(1.0 / publish_rate, self._publish_distances)
        self.create_timer(1.0 / 30.0, self._gui_tick)

        self.get_logger().info("keypoint_node ready, waiting for first frame...")

    # ------------------------------------------------------------------ subscribers

    def _color_cb(self, msg: CompressedImage):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        with self._lock:
            self._latest_color = img
            self._track_step(img)

    def _depth_cb(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        with self._lock:
            self._latest_depth = depth

    def _info_cb(self, msg: CameraInfo):
        with self._lock:
            self._fx, self._fy = msg.k[0], msg.k[4]
            self._ppx, self._ppy = msg.k[2], msg.k[5]

    # ------------------------------------------------------------------ tracking

    def _track_step(self, color_bgr: np.ndarray):
        if self._awaiting_clicks or self._track is None:
            self._prev_gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
            return

        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._track.pts, None, **self.lk_params)

        for i in range(len(self._track.pts)):
            if status[i][0] == 0 or new_pts[i][0][0] < 0 or new_pts[i][0][1] < 0:
                cx, cy = self._track.pts[i][0]
                rx, ry, sim = self.dino.search_best(
                    color_bgr, self._track.refs[i], int(cx), int(cy),
                    roi=self.recovery_roi, stride=self.recovery_stride,
                )
                if sim >= self.recovery_threshold:
                    new_pts[i][0] = (rx, ry)
                else:
                    new_pts[i][0] = self._track.pts[i][0]

        self._track.pts = new_pts.astype(np.float32)
        self._update_xyz()
        self._prev_gray = gray

    def _update_xyz(self):
        if self._track is None or self._latest_depth is None:
            return
        if self._fx is None:
            return

        depth = self._latest_depth
        h, w = depth.shape
        new_xyz = np.array(self._track.last_xyz, copy=True)
        new_valid = np.array(self._track.valid, copy=True)

        for i, (x, y) in enumerate(self._track.pts.reshape(-1, 2)):
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h):
                new_valid[i] = False
                continue
            d_raw = self._sample_depth(depth, xi, yi)
            if d_raw <= 0:
                new_valid[i] = False
                continue
            z = d_raw * self.depth_scale
            xm = (xi - self._ppx) * z / self._fx
            ym = (yi - self._ppy) * z / self._fy
            new_xyz[i] = (xm, ym, z)
            new_valid[i] = True

        self._track.last_xyz = new_xyz
        self._track.valid = new_valid

    def _sample_depth(self, depth: np.ndarray, x: int, y: int) -> float:
        v = depth[y, x]
        if v > 0:
            return float(v)
        h, w = depth.shape
        x0, x1 = max(0, x - 1), min(w, x + 2)
        y0, y1 = max(0, y - 1), min(h, y + 2)
        patch = depth[y0:y1, x0:x1]
        nz = patch[patch > 0]
        if nz.size == 0:
            return 0.0
        return float(np.median(nz))

    # ------------------------------------------------------------------ publishing

    def _publish_distances(self):
        if self._track is None:
            return
        d_01, d_12 = float("nan"), float("nan")
        if self._track.valid[0] and self._track.valid[1]:
            d_01 = float(np.linalg.norm(self._track.last_xyz[1] - self._track.last_xyz[0]))
        if self._track.valid[1] and self._track.valid[2]:
            d_12 = float(np.linalg.norm(self._track.last_xyz[2] - self._track.last_xyz[1]))
        if not np.isfinite(d_01) or not np.isfinite(d_12):
            return
        msg = Float32MultiArray()
        msg.data = [d_01, d_12]
        self.pub_distances.publish(msg)

    # ------------------------------------------------------------------ GUI

    def _gui_tick(self):
        with self._lock:
            color = None if self._latest_color is None else self._latest_color.copy()
        if color is None:
            return

        self._maybe_open_window()
        overlay = self._render(color)
        cv2.imshow(WINDOW_NAME, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._shutdown = True
            rclpy.shutdown()
        elif key == ord("r"):
            self._reset_clicks()

    def _maybe_open_window(self):
        if not self._awaiting_clicks and self._track is None:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
            self._awaiting_clicks = True
            self._click_buffer = []
            self.get_logger().info(
                f"Click {self.n_keypoints} keypoints in order: " + ", ".join(LABELS[:self.n_keypoints])
            )

    def _reset_clicks(self):
        self._track = None
        self._awaiting_clicks = False
        self._click_buffer = []
        self.get_logger().info("Reset — re-click keypoints.")

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or not self._awaiting_clicks:
            return
        self._click_buffer.append((x, y))
        self.get_logger().info(f"  [{len(self._click_buffer)}/{self.n_keypoints}] {LABELS[len(self._click_buffer)-1]} = ({x}, {y})")
        if len(self._click_buffer) == self.n_keypoints:
            self._finalise_clicks()

    def _finalise_clicks(self):
        with self._lock:
            color = None if self._latest_color is None else self._latest_color.copy()
        if color is None:
            return
        refs = [self.dino.embed(color, x, y) for x, y in self._click_buffer]
        pts = np.array(self._click_buffer, dtype=np.float32).reshape(-1, 1, 2)
        self._track = TrackState(
            pts=pts,
            refs=refs,
            last_xyz=np.zeros((self.n_keypoints, 3), dtype=np.float32),
            valid=np.zeros(self.n_keypoints, dtype=bool),
        )
        self._prev_gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        self._awaiting_clicks = False
        self._update_xyz()
        self.get_logger().info("Tracking started.")

    def _render(self, image: np.ndarray) -> np.ndarray:
        out = image.copy()
        if self._awaiting_clicks:
            cv2.putText(
                out,
                f"Click {self.n_keypoints} kps: {', '.join(LABELS[:self.n_keypoints])}  "
                f"({len(self._click_buffer)}/{self.n_keypoints})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            for i, (x, y) in enumerate(self._click_buffer):
                cv2.circle(out, (x, y), 6, DRAW_COLOURS[i], 2)
            return out

        if self._track is not None:
            for i, (x, y) in enumerate(self._track.pts.reshape(-1, 2)):
                colour = DRAW_COLOURS[i] if self._track.valid[i] else (128, 128, 128)
                cv2.circle(out, (int(x), int(y)), 6, colour, 2)
                cv2.putText(out, LABELS[i], (int(x) + 8, int(y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
            if self._track.valid[0] and self._track.valid[1]:
                d01 = np.linalg.norm(self._track.last_xyz[1] - self._track.last_xyz[0])
                cv2.putText(out, f"d(R,obj)={d01:.3f} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if self._track.valid[1] and self._track.valid[2]:
                d12 = np.linalg.norm(self._track.last_xyz[2] - self._track.last_xyz[1])
                cv2.putText(out, f"d(L,obj)={d12:.3f} m", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    def destroy_node(self):
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = KeypointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
