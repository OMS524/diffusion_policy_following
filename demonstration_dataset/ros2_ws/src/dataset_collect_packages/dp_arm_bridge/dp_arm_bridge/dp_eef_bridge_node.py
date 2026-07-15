#!/usr/bin/env python3
#
# Copyright 2025 AIRO LABS., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Read-only Piper CAN -> Diffusion Policy EEF telemetry bridge.

This is a separate superset data-collection node.  It leaves the ACT
``arm_bridge_node.py`` unchanged, republishes the same four joint-space topics,
and adds Cartesian end-effector observation/action topics for Diffusion Policy.

Published joint topics (``sensor_msgs/JointState``):

  /left_arm/joint_states   -- measured qpos
  /right_arm/joint_states  -- measured qpos
  /left_arm/joint_ctrl     -- commanded joint-space action
  /right_arm/joint_ctrl    -- commanded joint-space action

Published EEF topics (``std_msgs/Float64MultiArray``):

  /dp/eef_actual  -- measured observation from feedback FK + gripper feedback
  /dp/eef_target  -- commanded action from control FK + gripper control target

Both messages contain the same fixed 14-value layout::

  [left_x, left_y, left_z,
   left_rotvec_x, left_rotvec_y, left_rotvec_z, left_gripper_width,
   right_x, right_y, right_z,
   right_rotvec_x, right_rotvec_y, right_rotvec_z, right_gripper_width]

Units are metres for position/gripper width and radians for the rotation
vector.  Each arm pose is relative to that arm's Piper ``base_link``.  The SDK
FK reports fixed-axis XYZ Euler angles; this node converts them to a rotation
vector before publishing.

``/dp/eef_actual`` maps to the proprioceptive observation.  The distinct
``/dp/eef_target`` topic maps to the Diffusion Policy action and must not be
replaced with the actual pose.

SAFETY: the node is read-only by default.  It does not enable motors or send
motion commands.  ``EnableFkCal`` only enables SDK-side FK computation.
``MasterSlaveConfig`` remains opt-in because it sends a CAN configuration
instruction that can change the arm's master/slave role.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


VECTOR_SIZE = 14
VECTOR_LAYOUT = (
    "left[x_m,y_m,z_m,rx_rad,ry_rad,rz_rad,grip_m],"
    "right[x_m,y_m,z_m,rx_rad,ry_rad,rz_rad,grip_m]"
)


def fk_position_mm_to_m(value: float) -> float:
    """Piper SDK GetFK XYZ value in millimetres -> metres."""
    return float(value) * 1e-3


def raw_joint_angle_to_rad(raw: int) -> float:
    """Piper CAN joint value in 0.001 degrees -> radians."""
    return math.radians(float(raw) * 0.001)


def fk_angle_deg_to_rad(value: float) -> float:
    """Piper SDK GetFK Euler angle in degrees -> radians."""
    return math.radians(float(value))


def raw_grip_to_m(raw: int) -> float:
    """Piper 0.001-mm gripper stroke integer -> metres."""
    return float(raw) * 1e-6


def euler_xyz_to_rotvec(rx: float, ry: float, rz: float):
    """Fixed-axis XYZ Euler angles -> shortest axis-angle rotation vector.

    Piper SDK describes its Euler conversion as extrinsic/static XYZ (sxyz).
    The equivalent rotation matrix order is Rz(rz) @ Ry(ry) @ Rx(rx).
    """
    half_rx = 0.5 * rx
    half_ry = 0.5 * ry
    half_rz = 0.5 * rz
    cr, sr = math.cos(half_rx), math.sin(half_rx)
    cp, sp = math.cos(half_ry), math.sin(half_ry)
    cy, sy = math.cos(half_rz), math.sin(half_rz)

    # Unit quaternion in (x, y, z, w) order.
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0]
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # q and -q encode the same rotation.  Choose qw >= 0 so the resulting
    # rotation-vector magnitude stays in the shortest [0, pi] range.
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw

    sin_half = math.sqrt(qx * qx + qy * qy + qz * qz)
    if sin_half <= 1e-12:
        return [0.0, 0.0, 0.0]

    angle = 2.0 * math.atan2(sin_half, max(-1.0, min(1.0, qw)))
    scale = angle / sin_half
    return [qx * scale, qy * scale, qz * scale]


def make_vector_msg(values):
    """Build a labelled 14-value Float64MultiArray."""
    if len(values) != VECTOR_SIZE:
        raise ValueError(f"expected {VECTOR_SIZE} EEF values, got {len(values)}")

    dim = MultiArrayDimension()
    dim.label = VECTOR_LAYOUT
    dim.size = VECTOR_SIZE
    dim.stride = VECTOR_SIZE

    msg = Float64MultiArray()
    msg.layout.dim = [dim]
    msg.layout.data_offset = 0
    msg.data = [float(value) for value in values]
    return msg


class _ArmReader:
    """Read one Piper interface and build joint plus Cartesian telemetry."""

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
            self._iface.MasterSlaveConfig(0xFC, 0, 0, 0)
            node.get_logger().warn(
                f"[{ns}] MasterSlaveConfig(0xFC) issued on {can_port} "
                "(enable_master_slave_config=true)"
            )

        try:
            self._iface.EnableFkCal()
        except AttributeError as exc:
            raise RuntimeError(
                "installed piper_sdk does not provide EnableFkCal/GetFK; "
                "install piper_sdk 0.3.0 or newer"
            ) from exc

        # Give the SDK receive/FK threads time to replace their initial zeros.
        time.sleep(0.1)
        self._prev_pos = None
        self._prev_t = None
        node.get_logger().info(
            f"[{ns}] connected on {can_port}; SDK feedback/control FK enabled"
        )

    def _measured(self):
        """Measured [j1..j6 rad, gripper m] from arm feedback."""
        joints = self._iface.GetArmJointMsgs().joint_state
        grip = self._iface.GetArmGripperMsgs().gripper_state.grippers_angle
        return [
            raw_joint_angle_to_rad(joints.joint_1),
            raw_joint_angle_to_rad(joints.joint_2),
            raw_joint_angle_to_rad(joints.joint_3),
            raw_joint_angle_to_rad(joints.joint_4),
            raw_joint_angle_to_rad(joints.joint_5),
            raw_joint_angle_to_rad(joints.joint_6),
            raw_grip_to_m(grip),
        ]

    def _commanded(self):
        """Commanded [j1..j6 rad, gripper m] from leader control frames."""
        joints = self._iface.GetArmJointCtrl().joint_ctrl
        grip = self._iface.GetArmGripperCtrl().gripper_ctrl.grippers_angle
        return [
            raw_joint_angle_to_rad(joints.joint_1),
            raw_joint_angle_to_rad(joints.joint_2),
            raw_joint_angle_to_rad(joints.joint_3),
            raw_joint_angle_to_rad(joints.joint_4),
            raw_joint_angle_to_rad(joints.joint_5),
            raw_joint_angle_to_rad(joints.joint_6),
            raw_grip_to_m(grip),
        ]

    def _fk_pose(self, mode: str):
        """Return [x,y,z,rotvec_x,rotvec_y,rotvec_z] for joint 6."""
        fk = self._iface.GetFK(mode)
        if fk is None or len(fk) < 6 or len(fk[-1]) < 6:
            raise RuntimeError(f"invalid GetFK({mode!r}) result: {fk!r}")

        # GetFK differs from the raw CAN messages: XYZ is already expressed in
        # millimetres and the Euler angles are already expressed in degrees.
        # Do not apply the CAN protocol's additional 0.001 scale here.
        raw = fk[-1]
        position = [fk_position_mm_to_m(raw[i]) for i in range(3)]
        euler = [fk_angle_deg_to_rad(raw[i]) for i in range(3, 6)]
        rotvec = euler_xyz_to_rotvec(*euler)
        return position + rotvec

    def make_msgs(self, stamp):
        """Return joint messages and aligned actual/target EEF vectors."""
        measured = self._measured()
        commanded = self._commanded()

        now = time.monotonic()
        if self._prev_pos is None or self._prev_t is None:
            velocity = [0.0] * 6
        else:
            dt = max(now - self._prev_t, 1e-3)
            velocity = [
                (measured[i] - self._prev_pos[i]) / dt for i in range(6)
            ]
        self._prev_pos, self._prev_t = measured, now

        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.name = self.JOINT_NAMES
        joint_state.position = measured
        joint_state.velocity = velocity
        joint_state.effort = [0.0] * 7

        joint_ctrl = JointState()
        joint_ctrl.header.stamp = stamp
        joint_ctrl.name = self.JOINT_NAMES
        joint_ctrl.position = commanded

        # Reuse the exact gripper samples placed in the joint messages so the
        # joint and Cartesian representations describe the same timer sample.
        actual = self._fk_pose("feedback") + [measured[6]]
        target = self._fk_pose("control") + [commanded[6]]
        return joint_state, joint_ctrl, actual, target

    def feedback_is_zero(self) -> bool:
        return all(abs(value) < 1e-9 for value in self._measured()[:6])


class DpEefBridgeNode(Node):
    """Publish the ACT joint schema plus synchronized DP EEF vectors."""

    def __init__(self):
        super().__init__("dp_eef_bridge_node")
        self.declare_parameter("left_can", "can_left")
        self.declare_parameter("right_can", "can_right")
        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("enable_master_slave_config", False)

        left_can = str(self.get_parameter("left_can").value)
        right_can = str(self.get_parameter("right_can").value)
        rate = float(self.get_parameter("rate_hz").value)
        ms_cfg = bool(self.get_parameter("enable_master_slave_config").value)
        if rate <= 0.0:
            raise ValueError(f"rate_hz must be positive, got {rate}")

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self._arms = []
        for can_port, ns in ((left_can, "left_arm"),
                             (right_can, "right_arm")):
            reader = _ArmReader(self, can_port, ns, ms_cfg)
            state_pub = self.create_publisher(
                JointState, f"/{ns}/joint_states", qos)
            ctrl_pub = self.create_publisher(
                JointState, f"/{ns}/joint_ctrl", qos)
            self._arms.append((reader, state_pub, ctrl_pub))

        self._actual_pub = self.create_publisher(
            Float64MultiArray, "/dp/eef_actual", qos)
        self._target_pub = self.create_publisher(
            Float64MultiArray, "/dp/eef_target", qos)

        for reader, _, _ in self._arms:
            if reader.feedback_is_zero():
                self.get_logger().warn(
                    f"[{reader.ns}] joint feedback reads ZERO on "
                    f"{reader.can_port}. Check CAN wiring, arm power, slave "
                    "mode, and SDK version."
                )

        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"dp_eef_bridge publishing joint states/control + 14D EEF "
            f"actual/target @ "
            f"{rate:.0f} Hz (left={left_can}, right={right_can})"
        )

    def _tick(self):
        stamp = self.get_clock().now().to_msg()
        actual = []
        target = []
        for reader, state_pub, ctrl_pub in self._arms:
            try:
                joint_state, joint_ctrl, arm_actual, arm_target = \
                    reader.make_msgs(stamp)
            except Exception as exc:  # noqa: BLE001 - SDK can raise broadly
                self.get_logger().warn(f"[{reader.ns}] read failed: {exc}")
                continue

            state_pub.publish(joint_state)
            ctrl_pub.publish(joint_ctrl)
            actual.extend(arm_actual)
            target.extend(arm_target)

        # Do not publish an incomplete dual-arm EEF vector if one arm failed.
        if len(actual) != VECTOR_SIZE or len(target) != VECTOR_SIZE:
            return

        actual_msg = make_vector_msg(actual)
        target_msg = make_vector_msg(target)

        # Both are emitted from the same timer callback so their rosbag receive
        # times stay adjacent.  A custom stamped message can replace this array
        # type later without changing the actual/target data semantics.
        self._actual_pub.publish(actual_msg)
        self._target_pub.publish(target_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DpEefBridgeNode()
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
