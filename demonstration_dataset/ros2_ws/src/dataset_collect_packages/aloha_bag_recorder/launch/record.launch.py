"""Launch the aloha_recorder node with parameters from config/recorder.yaml.

The yaml file is the source of truth; launch arguments override individual
values only when set non-empty, e.g.:

    ros2 launch aloha_bag_recorder record.launch.py save_dir:=/data/raw task_name:=pick_place
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _make_node(context, *args, **kwargs):
    pkg_share = get_package_share_directory("aloha_bag_recorder")
    config = LaunchConfiguration("config").perform(context) or \
        os.path.join(pkg_share, "config", "recorder.yaml")

    # yaml first; then only the overrides the user actually set (non-empty),
    # so empty launch args never clobber the yaml values.
    parameters = [config]
    overrides = {}
    for key in ("save_dir", "task_name"):
        value = LaunchConfiguration(key).perform(context)
        if value:
            overrides[key] = value
    # numeric overrides need explicit typing (perform() returns a string)
    dur = LaunchConfiguration("record_duration").perform(context)
    if dur:
        overrides["record_duration"] = float(dur)
    frames = LaunchConfiguration("record_frames").perform(context)
    if frames:
        overrides["record_frames"] = int(frames)
    topic = LaunchConfiguration("frame_count_topic").perform(context)
    if topic:
        overrides["frame_count_topic"] = topic
    if overrides:
        parameters.append(overrides)

    return [Node(
        package="aloha_bag_recorder",
        executable="recorder_node",
        name="aloha_recorder",
        output="screen",
        parameters=parameters,
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "config", default_value="",
            description="Path to the recorder parameter yaml (empty = packaged default)."),
        DeclareLaunchArgument(
            "save_dir", default_value="",
            description="Override save_dir (empty keeps the yaml value)."),
        DeclareLaunchArgument(
            "task_name", default_value="",
            description="Override task_name sub-directory (empty keeps the yaml value)."),
        DeclareLaunchArgument(
            "record_duration", default_value="",
            description="Override auto-stop seconds (empty keeps the yaml value; <=0 = manual)."),
        DeclareLaunchArgument(
            "record_frames", default_value="",
            description="Override exact-frame auto-stop count (empty keeps the yaml value; <=0 = off)."),
        DeclareLaunchArgument(
            "frame_count_topic", default_value="",
            description="Override the counted reference topic (empty keeps the yaml value)."),
        OpaqueFunction(function=_make_node),
    ])
