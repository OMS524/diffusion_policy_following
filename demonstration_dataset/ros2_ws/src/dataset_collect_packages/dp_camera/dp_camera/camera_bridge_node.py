#!/usr/bin/env python3
#
# Copyright 2025 AIRO LABS., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""RealSense D435i bridge: publishes JPEG-compressed RGB and, optionally, aligned depth + camera_info.

Each instance owns one RealSense device, selected by librealsense serial number
(`serial` parameter) or, when serial is empty, by USB physical-port suffix
(`usb_port` parameter, e.g. "2.4.1"). All topics are published under the
node namespace, so launching with `namespace=cam_top` yields:

  /cam_top/image_compressed   sensor_msgs/CompressedImage           (always)
  /cam_top/depth_raw          sensor_msgs/Image (16UC1, millimetres) (only when enable_depth)
  /cam_top/camera_info        sensor_msgs/CameraInfo                 (only when enable_depth)

`enable_depth` (bool parameter, default True) gates the depth stream. When False
the device runs colour-only: no depth stream is requested from librealsense, no
`rs.align` is performed, and neither depth_raw nor camera_info is advertised.
This is how the wrist cameras (cam_left / cam_right) stay RGB-only so the shared
USB controller is not saturated, while cam_top / cam_front stream aligned depth.

Note on intrinsics: depth is aligned *into the colour frame* (`rs.align(color)`),
so the published depth shares the colour image's resolution and pixel grid. The
correct intrinsics for the aligned depth are therefore the COLOUR stream's
intrinsics (used here) — not the raw depth stream's own intrinsics.
"""

import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


def _select_device(serial: str, usb_port: str) -> rs.device:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        raise RuntimeError("No RealSense devices found")

    if serial:
        for d in devices:
            if d.get_info(rs.camera_info.serial_number) == serial:
                return d
        raise RuntimeError(f"RealSense with serial '{serial}' not found")

    if usb_port:
        for d in devices:
            phys = d.get_info(rs.camera_info.physical_port)
            if phys.endswith(f"{usb_port}/video4linux") or f"-{usb_port}/" in phys or phys.endswith(usb_port):
                return d
        raise RuntimeError(f"RealSense at USB port '{usb_port}' not found")

    raise RuntimeError("Either 'serial' or 'usb_port' parameter must be provided")


class CameraBridgeNode(Node):
    def __init__(self):
        super().__init__("camera_bridge_node")

        self.declare_parameter("serial", "")
        self.declare_parameter("usb_port", "")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("enable_depth", True)

        serial = self.get_parameter("serial").value
        usb_port = self.get_parameter("usb_port").value
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = int(self.get_parameter("fps").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.enable_depth = bool(self.get_parameter("enable_depth").value)

        device = _select_device(serial, usb_port)
        device_serial = device.get_info(rs.camera_info.serial_number)
        self.get_logger().info(
            f"Opening RealSense serial={device_serial} "
            f"port={device.get_info(rs.camera_info.physical_port)}"
        )

        config = rs.config()
        config.enable_device(device_serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        # pipeline.start() intermittently fails with "Device or resource busy"
        # when several RealSense cameras initialise at once (a transient race as
        # sibling nodes claim USB resources). A gentle retry rides over it
        # without disturbing the other cameras (a hardware_reset here would knock
        # the siblings out). If it never succeeds, the device is genuinely held
        # by another process — surface that by letting the node fail.
        profile = None
        last_err = None
        for attempt in range(1, 9):
            self.pipeline = rs.pipeline()
            try:
                profile = self.pipeline.start(config)
                break
            except RuntimeError as e:
                last_err = e
                self.get_logger().warn(
                    f"pipeline.start() failed (attempt {attempt}/8): {e}; retrying in 2s")
                time.sleep(2.0)
        if profile is None:
            raise RuntimeError(f"pipeline.start() failed after 8 attempts: {last_err}")
        self.bridge = CvBridge()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub_color = self.create_publisher(CompressedImage, "image_compressed", sensor_qos)

        if self.enable_depth:
            self.align = rs.align(rs.stream.color)
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            # Aligned depth lives on the colour grid, so use the colour intrinsics.
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.intrinsics = color_profile.get_intrinsics()
            self.cached_camera_info = self._build_camera_info(self.intrinsics)
            self.pub_depth = self.create_publisher(Image, "depth_raw", sensor_qos)
            self.pub_info = self.create_publisher(CameraInfo, "camera_info", sensor_qos)
            self.get_logger().info(
                f"depth ENABLED (depth_scale={self.depth_scale:.6f} m/unit); "
                "publishing depth_raw + camera_info"
            )
        else:
            self.align = None
            self.depth_scale = None
            self.intrinsics = None
            self.cached_camera_info = None
            self.pub_depth = None
            self.pub_info = None
            self.get_logger().info("depth DISABLED (colour-only); not advertising depth_raw/camera_info")

        self._running = True
        self._thread = threading.Thread(target=self._spin_camera, daemon=True)
        self._thread.start()

    def _build_camera_info(self, intr) -> CameraInfo:
        info = CameraInfo()
        info.width = intr.width
        info.height = intr.height
        info.distortion_model = "plumb_bob"
        info.d = list(intr.coeffs) + [0.0] * max(0, 5 - len(intr.coeffs))
        info.k = [intr.fx, 0.0, intr.ppx,
                  0.0, intr.fy, intr.ppy,
                  0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.p = [intr.fx, 0.0, intr.ppx, 0.0,
                  0.0, intr.fy, intr.ppy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        return info

    def _spin_camera(self):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError as e:
                self.get_logger().warning(f"wait_for_frames failed: {e}")
                continue

            if self.enable_depth:
                frames = self.align.process(frames)
            color = frames.get_color_frame()
            if not color:
                continue
            depth = frames.get_depth_frame() if self.enable_depth else None
            if self.enable_depth and not depth:
                continue

            stamp = self.get_clock().now().to_msg()

            color_np = np.asanyarray(color.get_data())
            ok, buf = cv2.imencode(".jpg", color_np, encode_params)
            if not ok:
                continue
            color_msg = CompressedImage()
            color_msg.header.stamp = stamp
            color_msg.header.frame_id = self.frame_id
            color_msg.format = "jpeg"
            color_msg.data = buf.tobytes()
            self.pub_color.publish(color_msg)

            if self.enable_depth:
                depth_np = np.asanyarray(depth.get_data()).astype(np.uint16)
                depth_msg = self.bridge.cv2_to_imgmsg(depth_np, encoding="16UC1")
                depth_msg.header.stamp = stamp
                depth_msg.header.frame_id = self.frame_id

                info_msg = self.cached_camera_info
                info_msg.header.stamp = stamp
                info_msg.header.frame_id = self.frame_id

                self.pub_depth.publish(depth_msg)
                self.pub_info.publish(info_msg)

    def destroy_node(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = CameraBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
