#!/usr/bin/env python3
#
# Copyright 2025 AIRO LABS., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""All publishers needed for ACT data collection, in one launch.

Brings up:
  * the 4 RealSense camera bridges (dp_camera/cameras.launch.py)
    -> /cam_{top,left,front,right}/image_compressed
    -> RGB-only for DP collection (depth/camera_info disabled)
  * the dual-Piper CAN->ROS2 telemetry bridge (dp_arm_bridge/arm_bridge.launch.py)
    -> /{left,right}_arm/joint_states (qpos) + /{left,right}_arm/joint_ctrl (action)

PREREQUISITE: bring up CAN first (needs sudo):
    bash ~/piper_sdk/piper_sdk/can_activate.sh can_left  1000000 <usb_port>
    bash ~/piper_sdk/piper_sdk/can_activate.sh can_right 1000000 <usb_port>

This replaces the legacy aloha_piper.launch.py, whose `piper` driver is ROS1
and does not run under ROS 2. Record with the separate aloha_bag_recorder node.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description() -> LaunchDescription:
    cameras = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('dp_camera'), 'launch', 'cameras.launch.py')))
    arms = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('dp_arm_bridge'), 'launch', 'arm_bridge.launch.py')))
    return LaunchDescription([cameras, arms])
