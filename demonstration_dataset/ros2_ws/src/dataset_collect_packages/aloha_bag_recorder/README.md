# aloha_bag_recorder

Episode-wise rosbag2 (`.db3`) recorder for the airo / Piper ALOHA setup. One
`std_srvs/Trigger` service each starts / stops / discards an episode; every
received message is written to a per-episode bag under `save_dir`.

```bash
ros2 launch aloha_bag_recorder record.launch.py          # or run the node directly
ros2 service call /aloha_recorder/start_recording   std_srvs/srv/Trigger
ros2 service call /aloha_recorder/stop_recording    std_srvs/srv/Trigger
ros2 service call /aloha_recorder/discard_recording std_srvs/srv/Trigger
```

For Diffusion Policy collection, use the separate recorder executable. It
records the four compressed RGB streams, arm/TF data, `/dp/eef_actual`, and
`/dp/eef_target`. It deliberately excludes depth images and camera-info, and
exposes its services below `/dp_recorder`:

```bash
ros2 run aloha_bag_recorder dp_recorder_node --ros-args \
  -p save_dir:=./raw_data_dp \
  -p task_name:=place_yellow_cube_on_the_red_plate
ros2 service call /dp_recorder/start_recording std_srvs/srv/Trigger
ros2 service call /dp_recorder/stop_recording std_srvs/srv/Trigger
ros2 service call /dp_recorder/discard_recording std_srvs/srv/Trigger
```

## Recorded topics

Names match the `act_keypoint` `camera_bridge_node` namespaces
(`cam_top / cam_left / cam_front / cam_right`). See `topics.py`.

| Topic | Type | Notes |
|-------|------|-------|
| `/cam_{top,left,front,right}/image_compressed` | `CompressedImage` | JPEG, 4 cameras |
| `/cam_{top,front}/depth_raw` | `Image` (16UC1, mm) | **depth only on top + front** |
| `/cam_{top,front}/camera_info` | `CameraInfo` | intrinsics for depth back-projection |
| `/{left,right}_arm/joint_states` | `JointState` | measured → `qpos` |
| `/{left,right}_arm/joint_ctrl` | `JointState` | commanded (leader) → `/action` |
| `/tf` | `TFMessage` | |

The wrist cameras (`cam_left`, `cam_right`) are RGB-only by design: enabling
depth on all four cameras saturates the shared USB controller (all four
RealSense devices sit on a single USB root controller; depth runs ~28–29 Hz
under that load). Depth is limited to `cam_top` / `cam_front`.

## ⚠️ Bring publishers up BEFORE the first `start_recording`

Subscriptions are created lazily on the **first** `start_recording`, and each
subscription's QoS is negotiated from whatever publisher is live **at that
moment**, then frozen for the session. A best-effort sensor stream (colour /
depth) whose publisher is not up yet falls back to a default QoS that can leave
the recorder "subscribed but receiving zero messages".

Therefore:

1. Launch the camera nodes (`ros2 launch act_keypoint cameras.launch.py`) and
   the arm drivers first. Confirm `/cam_top/depth_raw` and `/cam_front/depth_raw`
   are actually publishing (`ros2 topic hz /cam_top/depth_raw`).
2. Only then call `start_recording`.

If any topic had no live publisher at that point, the node logs a single loud
`WARN` listing the missing topics — if you see depth topics there, stop, bring
the camera nodes up, and restart the recorder before collecting data.
