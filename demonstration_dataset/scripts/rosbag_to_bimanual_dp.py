#!/usr/bin/env python3
"""Convert dual-arm ROS 2 episode bags to a 10 Hz Diffusion Policy Zarr.

Input episode directories are produced by ``dp_recorder_node``.  The output is
a Zarr v2 group compatible with ``diffusion_policy.common.ReplayBuffer``:

    data/cam_top, cam_left, cam_front, cam_right  uint8 NHWC RGB
    data/eef_actual                               float32 [N, 14]
    data/eef_desired                              float32 [N, 14]
    data/action                                   float32 [N, 14]
    data/timestamp                                float64 [N]
    meta/episode_ends                             int64   [n_episodes]

All streams are aligned to an integer-nanosecond 10 Hz grid over their common
time range. Images, desired EEF observations, and actions use the latest sample
at or before each grid time (causal alignment). Actual EEF observation
translation/gripper values are linearly interpolated; rotations are
interpolated on SO(3) with quaternion SLERP before being returned as rotation
vectors.

``eef_desired`` and ``action`` intentionally contain the same desired command
stream on the replay-buffer time axis. The dataset SequenceSampler later uses
only the first ``n_obs_steps`` of observation keys while retaining the action
prediction horizon, matching Diffusion Policy's observation/action layout.
"""

import argparse
import glob
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
import zarr
from numcodecs import Blosc
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray


CAMERA_TOPICS = {
    "/cam_top/image_compressed": "cam_top",
    "/cam_left/image_compressed": "cam_left",
    "/cam_front/image_compressed": "cam_front",
    "/cam_right/image_compressed": "cam_right",
}
EEF_ACTUAL_TOPIC = "/dp/eef_actual"
EEF_TARGET_TOPIC = "/dp/eef_target"
REQUIRED_TOPICS = tuple(CAMERA_TOPICS) + (EEF_ACTUAL_TOPIC, EEF_TARGET_TOPIC)


def open_reader(episode_dir: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=episode_dir, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    return reader


def discover_episodes(input_dir: str):
    paths = sorted(glob.glob(os.path.join(input_dir, "episode_*")))
    episodes = [path for path in paths
                if glob.glob(os.path.join(path, "*.db3"))]
    if not episodes:
        raise FileNotFoundError(f"no episode_*/*.db3 under {input_dir}")
    return episodes


def read_timestamps_and_eef(episode_dir: str):
    """First pass: collect timestamps for all streams and deserialize only EEF."""
    timestamps = {topic: [] for topic in REQUIRED_TOPICS}
    actual_values = []
    target_values = []
    reader = open_reader(episode_dir)
    while reader.has_next():
        topic, payload, stamp_ns = reader.read_next()
        if topic not in timestamps:
            continue
        timestamps[topic].append(stamp_ns)
        if topic == EEF_ACTUAL_TOPIC:
            msg = deserialize_message(payload, Float64MultiArray)
            if len(msg.data) != 14:
                raise ValueError(
                    f"{episode_dir}: {topic} expected 14 values, got {len(msg.data)}")
            actual_values.append(msg.data)
        elif topic == EEF_TARGET_TOPIC:
            msg = deserialize_message(payload, Float64MultiArray)
            if len(msg.data) != 14:
                raise ValueError(
                    f"{episode_dir}: {topic} expected 14 values, got {len(msg.data)}")
            target_values.append(msg.data)

    missing = [topic for topic, values in timestamps.items() if not values]
    if missing:
        raise ValueError(f"{episode_dir}: missing required topic data: {missing}")

    timestamp_arrays = {
        topic: np.asarray(values, dtype=np.int64)
        for topic, values in timestamps.items()
    }
    actual = np.asarray(actual_values, dtype=np.float64)
    target = np.asarray(target_values, dtype=np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(target).all():
        raise ValueError(f"{episode_dir}: EEF data contains NaN or Inf")
    return timestamp_arrays, actual, target


def make_time_grid(timestamps, output_hz: float):
    period_ns = int(round(1e9 / output_hz))
    first_common = max(values[0] for values in timestamps.values())
    last_common = min(values[-1] for values in timestamps.values())
    start_ns = ((first_common + period_ns - 1) // period_ns) * period_ns
    if start_ns > last_common:
        raise ValueError("required streams have no common time interval")
    return np.arange(start_ns, last_common + 1, period_ns, dtype=np.int64)


def latest_before_indices(source_ns, query_ns):
    idxs = np.searchsorted(source_ns, query_ns, side="right") - 1
    if np.any(idxs < 0):
        raise ValueError("query precedes first source sample")
    return idxs.astype(np.int64)


def interpolate_eef(source_ns, values, query_ns):
    """Interpolate two 7D base->EEF poses independently."""
    hi = np.searchsorted(source_ns, query_ns, side="left")
    hi = np.clip(hi, 1, len(source_ns) - 1)
    lo = hi - 1
    denom = (source_ns[hi] - source_ns[lo]).astype(np.float64)
    alpha = (query_ns - source_ns[lo]) / denom
    alpha = np.clip(alpha, 0.0, 1.0)

    output = np.empty((len(query_ns), 14), dtype=np.float64)
    for arm_start in (0, 7):
        pos_slice = slice(arm_start, arm_start + 3)
        rot_slice = slice(arm_start + 3, arm_start + 6)
        grip_idx = arm_start + 6

        output[:, pos_slice] = (
            values[lo, pos_slice] * (1.0 - alpha[:, None])
            + values[hi, pos_slice] * alpha[:, None]
        )
        output[:, grip_idx] = (
            values[lo, grip_idx] * (1.0 - alpha)
            + values[hi, grip_idx] * alpha
        )

        for row, (lo_idx, hi_idx, fraction) in enumerate(zip(lo, hi, alpha)):
            rotations = Rotation.from_rotvec(
                values[[lo_idx, hi_idx], rot_slice])
            output[row, rot_slice] = Slerp(
                [0.0, 1.0], rotations)([float(fraction)]).as_rotvec()[0]

    return output.astype(np.float32)


def decode_selected_images(episode_dir, selected_indices, width, height):
    """Second pass: decode only the camera frames selected for the 10 Hz grid."""
    n_steps = len(next(iter(selected_indices.values())))
    images = {
        key: np.empty((n_steps, height, width, 3), dtype=np.uint8)
        for key in CAMERA_TOPICS.values()
    }
    source_ordinals = {topic: 0 for topic in CAMERA_TOPICS}
    wanted_rows = {
        topic: {int(source_idx): [] for source_idx in np.unique(idxs)}
        for topic, idxs in selected_indices.items()
    }
    for topic, idxs in selected_indices.items():
        for output_row, source_idx in enumerate(idxs):
            wanted_rows[topic][int(source_idx)].append(output_row)

    reader = open_reader(episode_dir)
    while reader.has_next():
        topic, payload, _ = reader.read_next()
        if topic not in CAMERA_TOPICS:
            continue
        ordinal = source_ordinals[topic]
        source_ordinals[topic] += 1
        rows = wanted_rows[topic].get(ordinal)
        if rows is None:
            continue

        msg = deserialize_message(payload, CompressedImage)
        encoded = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"{episode_dir}: failed to decode {topic} frame {ordinal}")
        resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        images[CAMERA_TOPICS[topic]][rows] = rgb

    return images


def create_output(path, height, width, overwrite):
    output = Path(path).expanduser().resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"output exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)

    root = zarr.open_group(str(output), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = Blosc(
        cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    for key in CAMERA_TOPICS.values():
        data.create_dataset(
            key, shape=(0, height, width, 3),
            chunks=(1, height, width, 3), dtype=np.uint8,
            compressor=compressor,
        )
    for key in ("eef_actual", "eef_desired", "action"):
        data.create_dataset(
            key, shape=(0, 14), chunks=(256, 14), dtype=np.float32,
            compressor=compressor,
        )
    data.create_dataset(
        "timestamp", shape=(0,), chunks=(1024,), dtype=np.float64,
        compressor=compressor,
    )
    meta.create_dataset(
        "episode_ends", shape=(0,), chunks=(64,), dtype=np.int64,
        compressor=None,
    )
    return output, root


def append_episode(root, episode_data):
    arrays = root["data"]
    old_size = arrays["action"].shape[0]
    episode_size = episode_data["action"].shape[0]
    new_size = old_size + episode_size
    for key, values in episode_data.items():
        array = arrays[key]
        array.resize((new_size,) + array.shape[1:])
        array[old_size:new_size] = values

    ends = root["meta"]["episode_ends"]
    ends.resize((ends.shape[0] + 1,))
    ends[-1] = new_size


def convert_episode(episode_dir, output_hz, width, height):
    timestamps, actual, target = read_timestamps_and_eef(episode_dir)
    grid_ns = make_time_grid(timestamps, output_hz)

    camera_indices = {
        topic: latest_before_indices(timestamps[topic], grid_ns)
        for topic in CAMERA_TOPICS
    }
    images = decode_selected_images(
        episode_dir, camera_indices, width=width, height=height)
    eef_actual = interpolate_eef(
        timestamps[EEF_ACTUAL_TOPIC], actual, grid_ns)
    target_indices = latest_before_indices(
        timestamps[EEF_TARGET_TOPIC], grid_ns)
    eef_desired = target[target_indices].astype(np.float32)
    action = eef_desired.copy()

    max_camera_age_ms = max(
        float(np.max(grid_ns - timestamps[topic][idxs])) / 1e6
        for topic, idxs in camera_indices.items()
    )
    max_target_age_ms = float(np.max(
        grid_ns - timestamps[EEF_TARGET_TOPIC][target_indices])) / 1e6
    episode_data = {
        **images,
        "eef_actual": eef_actual,
        "eef_desired": eef_desired,
        "action": action,
        "timestamp": grid_ns.astype(np.float64) / 1e9,
    }
    stats = {
        "steps": len(grid_ns),
        "duration": (grid_ns[-1] - grid_ns[0]) / 1e9 if len(grid_ns) > 1 else 0.0,
        "max_camera_age_ms": max_camera_age_ms,
        "max_target_age_ms": max_target_age_ms,
    }
    return episode_data, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="task directory containing episode_* bags")
    parser.add_argument("--output", required=True,
                        help="output Zarr directory")
    parser.add_argument("--output-hz", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_hz <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("output-hz, width, and height must be positive")

    episodes = discover_episodes(os.path.expanduser(args.input))
    output, root = create_output(
        args.output, height=args.height, width=args.width,
        overwrite=args.overwrite)
    root.attrs.update({
        "format": "bimanual_diffusion_policy_v2",
        "output_hz": float(args.output_hz),
        "image_layout": "NHWC_RGB_uint8",
        "image_size": [int(args.height), int(args.width)],
        "eef_layout": (
            "left[x,y,z,rx,ry,rz,gripper],"
            "right[x,y,z,rx,ry,rz,gripper]"
        ),
        "eef_units": "position/gripper=m, rotation_vector=rad",
        "action_source": EEF_TARGET_TOPIC,
        "actual_observation_source": EEF_ACTUAL_TOPIC,
        "desired_observation_source": EEF_TARGET_TOPIC,
        "target_alignment": "latest sample at or before each output timestamp",
        "observation_keys": [
            "cam_top", "cam_left", "cam_front", "cam_right",
            "eef_actual", "eef_desired",
        ],
    })

    print(f"converting {len(episodes)} episode(s) -> {output}")
    for episode_index, episode_dir in enumerate(episodes):
        episode_data, stats = convert_episode(
            episode_dir, output_hz=args.output_hz,
            width=args.width, height=args.height)
        append_episode(root, episode_data)
        print(
            f"[{episode_index + 1}/{len(episodes)}] "
            f"{os.path.basename(episode_dir)}: {stats['steps']} steps, "
            f"{stats['duration']:.1f}s, "
            f"camera_age<={stats['max_camera_age_ms']:.1f}ms, "
            f"target_age<={stats['max_target_age_ms']:.1f}ms"
        )

    n_steps = int(root["meta"]["episode_ends"][-1])
    print(
        f"done: {len(episodes)} episodes, {n_steps} steps, "
        f"{args.output_hz:g} Hz, RGB {args.width}x{args.height}"
    )


if __name__ == "__main__":
    main()
