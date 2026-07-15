#!/usr/bin/env python3
#
# Copyright 2025 AIRO LABS., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Launch the read-only dual-Piper CAN->ROS2 telemetry bridge."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    params = os.path.join(
        get_package_share_directory('dp_arm_bridge'), 'config', 'arms.yaml')
    return LaunchDescription([
        Node(
            package='dp_arm_bridge',
            executable='dp_eef_bridge_node',
            name='dp_eef_bridge_node',
            output='screen',
            parameters=[params],
        ),
    ])
