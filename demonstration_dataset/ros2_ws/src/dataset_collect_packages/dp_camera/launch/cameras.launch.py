#!/usr/bin/env python3
#
# Copyright 2025 AIRO LABS., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Launch one camera_bridge_node per RealSense camera listed in config/cameras.yaml."""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config_path = os.path.join(
        get_package_share_directory('dp_camera'),
        'config',
        'cameras.yaml',
    )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    stream = config.get('stream', {})
    width = int(stream.get('width', 640))
    height = int(stream.get('height', 480))
    fps = int(stream.get('fps', 30))
    jpeg_quality = int(stream.get('jpeg_quality', 90))

    actions = []
    for cam in config.get('cameras', []):
        actions.append(Node(
            package='dp_camera',
            executable='camera_bridge_node',
            name='camera_bridge_node',
            namespace=cam['namespace'],
            output='screen',
            parameters=[{
                'serial': cam.get('serial', ''),
                'usb_port': cam.get('usb_port', ''),
                'width': width,
                'height': height,
                'fps': fps,
                'jpeg_quality': jpeg_quality,
                'frame_id': f"{cam['namespace']}_color_optical_frame",
                'enable_depth': bool(cam.get('depth', False)),
            }],
        ))

    return LaunchDescription(actions)
