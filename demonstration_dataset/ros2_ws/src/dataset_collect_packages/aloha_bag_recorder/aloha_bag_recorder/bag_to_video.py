"""Render a recorded episode rosbag2 (.db3 + metadata.yaml) to an MP4 video.

Companion to ``recorder_node.py``: it reads the canonical 9-topic episode bags
produced by this package and composes the camera streams into a single video so
you can eyeball an episode without RViz or a full ROS replay.

What it shows
-------------
* the four RGB cameras (``/cam_*/image_compressed``, JPEG) as a 2x2 grid;
* the two depth streams (``/cam_top``/``/cam_front`` ``/depth_raw``, 16UC1 mm)
  colourised with a JET map, in a row below the RGB grid (toggle with --no-depth);
* a thin bottom strip with the per-arm joint + gripper values sampled at each
  rendered frame (toggle with --no-joints).

Why no cv_bridge
----------------
The system cv_bridge is built against NumPy 1.x and crashes under the NumPy 2.x
in the active conda env. We therefore decode JPEG with ``cv2.imdecode`` and parse
the raw 16UC1 depth buffer with ``numpy.frombuffer`` directly — neither needs
cv_bridge.

Usage
-----
    source /opt/ros/humble/setup.bash
    python3 bag_to_video.py <episode_dir>                 # -> <episode_dir>/<name>.mp4
    python3 bag_to_video.py <episode_dir> -o out.mp4
    python3 bag_to_video.py <task_dir> --batch            # every episode_* under it
    python3 bag_to_video.py <episode_dir> --no-depth --no-joints

``<episode_dir>`` is the directory holding ``*_0.db3`` and ``metadata.yaml``
(e.g. ``.../raw_data_depth/place_yellow_cube/episode_1781377935``).
"""

import argparse
import bisect
import glob
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - clearer message than a raw traceback
    sys.exit("cv2 (opencv-python) is required: pip install opencv-python")

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover
    sys.exit(f"ROS 2 not sourced ({exc}). Run: source /opt/ros/humble/setup.bash")


# Panel layout. RGB cameras fill a 2x2 grid in this order; depth cameras get
# their own row. Names match recorder topics_py / camera_bridge_node namespaces.
RGB_TOPICS = [
    "/cam_top/image_compressed",
    "/cam_front/image_compressed",
    "/cam_left/image_compressed",
    "/cam_right/image_compressed",
]
DEPTH_TOPICS = [
    "/cam_top/depth_raw",
    "/cam_front/depth_raw",
]
JOINT_TOPICS = [
    "/left_arm/joint_states",
    "/right_arm/joint_states",
]
# The reference stream the recorder counts (record_frames). Using it as the
# video timeline makes one rendered frame == one recorded base frame.
REFERENCE_TOPIC = "/cam_top/image_compressed"

FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- io
def find_bag(path):
    """Return (db3_dir, output_basename). Accepts an episode dir or a .db3 file."""
    if os.path.isfile(path) and path.endswith(".db3"):
        path = os.path.dirname(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    if not glob.glob(os.path.join(path, "*.db3")):
        raise FileNotFoundError(f"no .db3 in {path}")
    return path, os.path.basename(os.path.normpath(path))


def read_messages(bag_dir, wanted):
    """Yield (topic, msg, stamp_ns) for the wanted topics, in storage order."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_classes = {}
    for name in wanted:
        if name in type_map:
            msg_classes[name] = get_message(type_map[name])

    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        cls = msg_classes.get(topic)
        if cls is None:
            continue
        yield topic, deserialize_message(data, cls), stamp_ns


# ---------------------------------------------------------------- frame decode
def decode_jpeg(msg):
    """CompressedImage (JPEG) -> BGR uint8 array, or None on failure."""
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def decode_depth(msg, lo_mm, hi_mm):
    """16UC1 depth Image -> BGR colourised uint8 via JET over [lo_mm, hi_mm]."""
    dtype = np.dtype(np.uint16).newbyteorder(">" if msg.is_bigendian else "<")
    depth = np.frombuffer(bytes(msg.data), dtype=dtype)
    depth = depth.reshape(msg.height, msg.width).astype(np.float32)
    valid = depth > 0
    norm = np.clip((depth - lo_mm) / max(hi_mm - lo_mm, 1.0), 0.0, 1.0)
    norm8 = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(norm8, cv2.COLORMAP_JET)
    color[~valid] = (0, 0, 0)  # leave holes (no return) black, not dark-blue
    return color


def label(img, text):
    """Draw a small caption with a dark backing box in the top-left corner."""
    cv2.rectangle(img, (0, 0), (max(90, 9 * len(text) + 8), 22), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 16), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def placeholder(h, w, text):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.putText(img, text, (8, h // 2), FONT, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------- timeline
class Stream:
    """Decoded frames for one topic, kept sorted by timestamp for nearest lookup."""

    def __init__(self):
        self.stamps = []
        self.frames = []

    def add(self, stamp_ns, frame):
        if frame is not None:
            self.stamps.append(stamp_ns)
            self.frames.append(frame)

    def nearest(self, stamp_ns):
        if not self.stamps:
            return None
        i = bisect.bisect_left(self.stamps, stamp_ns)
        if i == 0:
            return self.frames[0]
        if i >= len(self.stamps):
            return self.frames[-1]
        before, after = self.stamps[i - 1], self.stamps[i]
        return self.frames[i - 1 if stamp_ns - before <= after - stamp_ns else i]


def joint_text(js):
    """Compact one-line summary of a JointState: 6 joints (deg) + gripper."""
    pos = list(js.position)
    if not pos:
        return "-"
    joints = " ".join(f"{np.degrees(p):6.1f}" for p in pos[:6])
    grip = f"{pos[6]:.3f}" if len(pos) > 6 else "-"
    return f"j[{joints}] g[{grip}]"


# ------------------------------------------------------------------- compose
def grid(panels, cols, panel_w, panel_h):
    """Tile resized panels row-major into `cols` columns; pad the last row."""
    if not panels:
        return None
    resized = [cv2.resize(p, (panel_w, panel_h)) for p in panels]
    rows = []
    for r in range(0, len(resized), cols):
        chunk = resized[r:r + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros((panel_h, panel_w, 3), np.uint8))
        rows.append(np.hstack(chunk))
    return np.vstack(rows)


def render(bag_dir, out_path, args):
    wanted = list(RGB_TOPICS)
    if not args.no_depth:
        wanted += DEPTH_TOPICS
    if not args.no_joints:
        wanted += JOINT_TOPICS

    streams = {t: Stream() for t in wanted}
    lo_mm, hi_mm = args.depth_range[0] * 1000.0, args.depth_range[1] * 1000.0

    for topic, msg, stamp in read_messages(bag_dir, wanted):
        if topic in RGB_TOPICS:
            streams[topic].add(stamp, decode_jpeg(msg))
        elif topic in DEPTH_TOPICS:
            streams[topic].add(stamp, decode_depth(msg, lo_mm, hi_mm))
        elif topic in JOINT_TOPICS:
            streams[topic].add(stamp, msg)  # keep the message; format at render

    ref = streams.get(REFERENCE_TOPIC)
    if ref is None or not ref.stamps:
        # fall back to whichever RGB stream has the most frames
        ref = max((streams[t] for t in RGB_TOPICS), key=lambda s: len(s.stamps))
    if not ref.stamps:
        print(f"  ! no camera frames in {bag_dir}, skipping")
        return False

    # FPS from the reference stream's own timing, unless overridden.
    if args.fps:
        fps = args.fps
    else:
        span_s = (ref.stamps[-1] - ref.stamps[0]) / 1e9
        fps = (len(ref.stamps) - 1) / span_s if span_s > 0 else 30.0
        fps = float(np.clip(fps, 1.0, 120.0))

    ph = args.panel_height
    pw = int(round(ph * 4 / 3))  # 4:3 cameras (640x480)

    writer = None
    n = len(ref.stamps)
    for k, stamp in enumerate(ref.stamps):
        rgb_panels = []
        for t in RGB_TOPICS:
            f = streams[t].nearest(stamp)
            name = t.split("/")[1]
            rgb_panels.append(label(f.copy(), name) if f is not None
                              else placeholder(ph, pw, name + " (none)"))
        canvas = grid(rgb_panels, 2, pw, ph)

        if not args.no_depth:
            depth_panels = []
            for t in DEPTH_TOPICS:
                f = streams[t].nearest(stamp)
                name = t.split("/")[1] + " depth"
                depth_panels.append(label(f.copy(), name) if f is not None
                                    else placeholder(ph, pw, name + " (none)"))
            depth_row = grid(depth_panels, 2, pw, ph)
            canvas = np.vstack([canvas, depth_row])

        if not args.no_joints:
            strip = np.zeros((26, canvas.shape[1], 3), np.uint8)
            parts = []
            for t in JOINT_TOPICS:
                js = streams[t].nearest(stamp)
                tag = "L" if "left" in t else "R"
                parts.append(f"{tag} {joint_text(js)}" if js is not None else f"{tag} -")
            cv2.putText(strip, "  ".join(parts), (4, 18), FONT, 0.4,
                        (180, 255, 180), 1, cv2.LINE_AA)
            canvas = np.vstack([canvas, strip])

        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"cannot open VideoWriter for {out_path}")
        writer.write(canvas)
        if k % 60 == 0:
            print(f"  {k + 1}/{n}", end="\r", flush=True)

    if writer is not None:
        writer.release()
    print(f"  wrote {out_path}  ({n} frames @ {fps:.1f} fps)")
    return True


# ----------------------------------------------------------------------- cli
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="episode dir (or .db3), or a task dir with --batch")
    p.add_argument("-o", "--output", help="output mp4 (default: <episode>/<name>.mp4)")
    p.add_argument("--batch", action="store_true",
                   help="treat path as a parent dir and render every episode_* under it")
    p.add_argument("--fps", type=float, default=0.0,
                   help="force output fps (default: derive from reference camera timing)")
    p.add_argument("--panel-height", type=int, default=300,
                   help="per-camera panel height in px (default 300)")
    p.add_argument("--depth-range", type=float, nargs=2, default=(0.2, 1.5),
                   metavar=("LO_M", "HI_M"),
                   help="depth colourmap range in metres (default 0.2 1.5)")
    p.add_argument("--no-depth", action="store_true", help="omit the depth row")
    p.add_argument("--no-joints", action="store_true", help="omit the joint strip")
    args = p.parse_args(argv)

    if args.batch:
        eps = sorted(glob.glob(os.path.join(args.path, "episode_*")))
        eps = [e for e in eps if glob.glob(os.path.join(e, "*.db3"))]
        if not eps:
            sys.exit(f"no episode_*/*.db3 under {args.path}")
        print(f"batch: {len(eps)} episode(s)")
        for ep in eps:
            bag_dir, name = find_bag(ep)
            print(f"[{name}]")
            render(bag_dir, os.path.join(bag_dir, f"{name}.mp4"), args)
        return

    bag_dir, name = find_bag(args.path)
    out = args.output or os.path.join(bag_dir, f"{name}.mp4")
    render(bag_dir, out, args)


if __name__ == "__main__":
    main()
