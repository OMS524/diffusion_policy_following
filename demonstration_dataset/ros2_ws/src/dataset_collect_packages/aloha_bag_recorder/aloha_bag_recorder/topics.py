"""Topic specification for the airo / Piper ALOHA bag recorder.

These nine topics reproduce *exactly* the schema of the existing
``piper_act_data/.../aloha_dataset/raw_data/episode_*/`` rosbags, so the
bags produced by this recorder stay drop-in compatible with whatever
downstream `.db3 -> ACT HDF5` conversion is used.

Verified against a real bag:
  - JointState topics carry 7 values: [joint1..joint6, gripper]
  - per ACT 14-dim convention qpos = [L joint1..6, L gripper, R joint1..6, R gripper]
  - joint_states = measured (-> qpos), joint_ctrl = commanded (-> /action)
  - image topics are JPEG CompressedImage ("rgb8; jpeg compressed bgr8")
"""

from dataclasses import dataclass
from typing import Type

from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from sensor_msgs.msg import JointState, CompressedImage, Image, CameraInfo
from tf2_msgs.msg import TFMessage


@dataclass(frozen=True)
class TopicSpec:
    name: str
    msg_class: Type
    type_str: str            # rosbag2 type string, e.g. "sensor_msgs/msg/JointState"
    reliability: str         # "reliable" | "best_effort" : QoS fallback if no publisher found


# Colour topics, one per camera. Names match the act_keypoint camera_bridge_node
# namespaces (cam_top / cam_left / cam_front / cam_right). cam_left and cam_right
# are the wrist cameras and stay RGB-only.
IMAGE_TOPICS = [
    "/cam_top/image_compressed",
    "/cam_left/image_compressed",
    "/cam_front/image_compressed",
    "/cam_right/image_compressed",
]

# Aligned depth maps. Only the top and front cameras stream depth (the wrist
# cameras are RGB-only to stay within the shared USB controller's bandwidth).
# These are raw (uncompressed) sensor_msgs/Image carrying 16UC1 millimetre depth
# — NOT CompressedImage. The topic name is `depth_raw` (the name advertised by
# camera_bridge_node and consumed by act_keypoint/keypoint_node), NOT depth_image.
DEPTH_TOPICS = [
    "/cam_top/depth_raw",
    "/cam_front/depth_raw",
]

# Intrinsics for the depth-enabled cameras, needed to back-project depth to 3D
# at conversion time. Published alongside depth by camera_bridge_node.
CAMERA_INFO_TOPICS = [
    "/cam_top/camera_info",
    "/cam_front/camera_info",
]

JOINT_TOPICS = [
    "/left_arm/joint_states",
    "/right_arm/joint_states",
    "/left_arm/joint_ctrl",
    "/right_arm/joint_ctrl",
]

TF_TOPIC = "/tf"


def _spec(name, msg_class, reliability):
    type_str = f"{msg_class.__module__.split('.')[0]}/msg/{msg_class.__name__}"
    return TopicSpec(name=name, msg_class=msg_class, type_str=type_str,
                     reliability=reliability)


def default_topic_specs():
    """The canonical topic set, in a stable order.

    Cameras (color + depth + camera_info) default to best_effort (typical sensor
    QoS); joints and tf to reliable. The actual subscription QoS is auto-negotiated
    from the live publisher at record time (see recorder_node.resolve_qos);
    these are only the fallback when no publisher is up yet.
    """
    specs = [_spec(t, CompressedImage, "best_effort") for t in IMAGE_TOPICS]
    specs += [_spec(t, Image, "best_effort") for t in DEPTH_TOPICS]
    specs += [_spec(t, CameraInfo, "best_effort") for t in CAMERA_INFO_TOPICS]
    specs += [_spec(t, JointState, "reliable") for t in JOINT_TOPICS]
    specs += [_spec(TF_TOPIC, TFMessage, "reliable")]
    return specs


def fallback_qos(reliability: str, depth: int) -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=(QoSReliabilityPolicy.RELIABLE if reliability == "reliable"
                     else QoSReliabilityPolicy.BEST_EFFORT),
        durability=QoSDurabilityPolicy.VOLATILE,
    )
