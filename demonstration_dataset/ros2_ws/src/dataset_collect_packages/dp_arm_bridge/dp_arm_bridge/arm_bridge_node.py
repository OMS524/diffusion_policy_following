#!/usr/bin/env python3
#
# Copyright 2025 AIRO LABS., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Read-only CAN->ROS2 telemetry bridge for the dual Piper arms.

Replaces the old ROS1 (`rospy`) piper driver for *data collection*: it reads
each follower arm over CAN via ``piper_sdk`` and republishes the joint topics
that ``aloha_bag_recorder`` records and ``act_episode_convert`` consumes.

Published per arm namespace (default ``left_arm`` / ``right_arm``):

  <ns>/joint_states   sensor_msgs/JointState  -- MEASURED follower state -> qpos
       position = [j1..j6 (rad), gripper (m, 0..0.07)]
       velocity = [v1..v6 (rad/s)]   (finite-difference)
       effort   = [0]*7              (placeholder; not used by ACT training)
  <ns>/joint_ctrl     sensor_msgs/JointState  -- COMMANDED target (leader) -> /action
       position = [j1..j6 (rad), gripper (m, 0..0.07)]

The control target (``GetArmJointCtrl`` / ``GetArmGripperCtrl``, CAN 0x155-0x159)
is the leader command in the hardware leader-follower linkage, and is distinct
from the measured feedback (``GetArmJointMsgs``) — so action != qpos, which is the
whole reason this ROS2 path is preferred over the slave-only piper_act_demo.

SAFETY: this node is read-only. It does NOT enable motors, send motion, or
reconfigure the arm. ``MasterSlaveConfig`` is OFF by default; enable it
(``enable_master_slave_config:=true``) ONLY if joint feedback reads all zeros and
you have confirmed it is safe for your leader-follower wiring (it issues the
0xFC linkage instruction, which can affect the master/slave role).
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import JointState


def raw_deg_to_rad(raw: int) -> float:
    """0.001-degree integer -> radians."""
    return math.radians(raw * 0.001)


def raw_grip_to_m(raw: int) -> float:
    """0.001-mm gripper stroke -> metres (0 .. ~0.07)."""
    return raw * 1e-6


class _ArmReader:
    """Wraps one Piper interface and converts its CAN feedback to JointState."""

    JOINT_NAMES = [f"joint{i + 1}" for i in range(6)] + ["gripper"]

    def __init__(self, node: Node, can_port: str, ns: str,
                 enable_ms_config: bool):
        from piper_sdk import C_PiperInterface_V2

        self.node = node
        self.ns = ns
        self.can_port = can_port
        self._iface = C_PiperInterface_V2(can_port, judge_flag=False)
        self._iface.ConnectPort()
        time.sleep(0.1)
        if enable_ms_config:
            # Opt-in only: re-issues the 0xFC slave-linkage instruction. Off by
            # default so we never perturb an already-working hardware linkage.
            self._iface.MasterSlaveConfig(0xFC, 0, 0, 0)
            node.get_logger().warn(
                f"[{ns}] MasterSlaveConfig(0xFC) issued on {can_port} "
                "(enable_master_slave_config=true)"
            )

        self._prev_pos = None
        self._prev_t = None
        node.get_logger().info(f"[{ns}] connected on {can_port} (read-only)")

    def _measured(self):
        """Measured 7-vector [j1..j6 rad, gripper m] from arm feedback."""
        j = self._iface.GetArmJointMsgs().joint_state
        g = self._iface.GetArmGripperMsgs().gripper_state.grippers_angle
        return [
            raw_deg_to_rad(j.joint_1), raw_deg_to_rad(j.joint_2),
            raw_deg_to_rad(j.joint_3), raw_deg_to_rad(j.joint_4),
            raw_deg_to_rad(j.joint_5), raw_deg_to_rad(j.joint_6),
            raw_grip_to_m(g),
        ]

    def _commanded(self):
        """Commanded (leader) 7-vector [j1..j6 rad, gripper m] from control msgs."""
        j = self._iface.GetArmJointCtrl().joint_ctrl
        g = self._iface.GetArmGripperCtrl().gripper_ctrl.grippers_angle
        return [
            raw_deg_to_rad(j.joint_1), raw_deg_to_rad(j.joint_2),
            raw_deg_to_rad(j.joint_3), raw_deg_to_rad(j.joint_4),
            raw_deg_to_rad(j.joint_5), raw_deg_to_rad(j.joint_6),
            raw_grip_to_m(g),
        ]

    def make_msgs(self, stamp):
        pos = self._measured()

        # finite-difference velocity over the 6 arm joints (no gripper vel)
        now = time.monotonic()
        if self._prev_pos is None or self._prev_t is None:
            vel = [0.0] * 6
        else:
            dt = max(now - self._prev_t, 1e-3)
            vel = [(pos[i] - self._prev_pos[i]) / dt for i in range(6)]
        self._prev_pos, self._prev_t = pos, now

        js = JointState()
        js.header.stamp = stamp
        js.name = self.JOINT_NAMES
        js.position = pos
        js.velocity = vel
        js.effort = [0.0] * 7

        jc = JointState()
        jc.header.stamp = stamp
        jc.name = self.JOINT_NAMES
        jc.position = self._commanded()
        return js, jc

    def feedback_is_zero(self) -> bool:
        return all(abs(v) < 1e-9 for v in self._measured()[:6])


class ArmBridgeNode(Node):
    def __init__(self):
        super().__init__("arm_bridge_node")
        self.declare_parameter("left_can", "can_left")
        self.declare_parameter("right_can", "can_right")
        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("enable_master_slave_config", False)

        left_can = self.get_parameter("left_can").value
        right_can = self.get_parameter("right_can").value
        rate = float(self.get_parameter("rate_hz").value)
        ms_cfg = bool(self.get_parameter("enable_master_slave_config").value)

        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                         reliability=QoSReliabilityPolicy.RELIABLE)

        self._arms = []
        for can_port, ns in [(left_can, "left_arm"), (right_can, "right_arm")]:
            reader = _ArmReader(self, can_port, ns, ms_cfg)
            pub_state = self.create_publisher(JointState, f"/{ns}/joint_states", qos)
            pub_ctrl = self.create_publisher(JointState, f"/{ns}/joint_ctrl", qos)
            self._arms.append((reader, pub_state, pub_ctrl))

        # one-shot sanity check: warn if feedback is all-zero
        for reader, _, _ in self._arms:
            if reader.feedback_is_zero():
                self.get_logger().warn(
                    f"[{reader.ns}] joint feedback reads ZERO on {reader.can_port}. "
                    "Check CAN wiring/power; if this persists, the arm may need "
                    "MasterSlaveConfig — relaunch with enable_master_slave_config:=true "
                    "(only if safe for your leader-follower setup)."
                )

        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"arm_bridge publishing joint_states+joint_ctrl @ {rate:.0f} Hz "
            f"(left={left_can}, right={right_can})"
        )

    def _tick(self):
        stamp = self.get_clock().now().to_msg()
        for reader, pub_state, pub_ctrl in self._arms:
            try:
                js, jc = reader.make_msgs(stamp)
            except Exception as exc:  # noqa: BLE001 - SDK can raise broadly
                self.get_logger().warn(f"[{reader.ns}] read failed: {exc}")
                continue
            pub_state.publish(js)
            pub_ctrl.publish(jc)


def main(args=None):
    rclpy.init(args=args)
    node = ArmBridgeNode()
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
