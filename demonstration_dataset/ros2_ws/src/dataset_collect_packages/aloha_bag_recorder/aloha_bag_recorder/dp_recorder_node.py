"""DP-specific RGB episode recorder that preserves the original ACT recorder.

This executable reuses the proven episode lifecycle, rosbag writer, QoS
negotiation, and frame-count auto-stop implementation from ``recorder_node``.
Its camera schema contains the four compressed RGB images and deliberately
excludes depth/camera-info topics. It also records ``/dp/eef_actual`` and
``/dp/eef_target``. The original ``recorder_node`` executable and topic schema
remain unchanged.
"""

import sys

import rclpy

from .dp_topics import dp_topic_specs
from .recorder_node import RecorderNode


class DpRecorderNode(RecorderNode):
    """Record RGB, arm, TF, and DP EEF actual/target data without depth."""

    def __init__(self):
        super().__init__()
        # RecorderNode creates subscriptions lazily on the first start request,
        # so replacing the schema here is complete and safe.
        self._specs = dp_topic_specs()
        self.get_logger().info(
            "dp_recorder ready: RGB-only cameras + arm/EEF topics (no depth). "
            "Call ~/start_recording to begin an episode."
        )


def main(args=None):
    # Keep services separate from the legacy /aloha_recorder services without
    # changing recorder_node.py.  A user-supplied __node remap still wins when
    # it appears later on the command line.
    init_args = list(sys.argv if args is None else args)
    init_args[1:1] = ["--ros-args", "-r", "__node:=dp_recorder"]
    rclpy.init(args=init_args)
    node = DpRecorderNode()
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
