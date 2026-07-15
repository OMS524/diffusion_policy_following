#!/usr/bin/env python3
#
# Copyright 2025 AIRO LABS., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Bring up all 4 camera bridges plus the keypoint tracker that consumes cam_top."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    cameras_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('dp_camera'),
                'launch',
                'cameras.launch.py',
            )
        )
    )

    keypoint_node = Node(
        package='dp_camera',
        executable='keypoint_node',
        name='keypoint_node',
        output='screen',
    )

    return LaunchDescription([cameras_launch, keypoint_node])
