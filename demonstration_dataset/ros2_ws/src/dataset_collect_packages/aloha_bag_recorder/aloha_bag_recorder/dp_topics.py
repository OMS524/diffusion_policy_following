"""RGB-only camera and robot topic schema for Diffusion Policy collection.

The legacy ACT recorder keeps its original colour/depth/camera-info schema in
``topics.default_topic_specs``.  The DP recorder intentionally records only
the four compressed RGB camera streams together with arm and EEF data.
"""

from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

from .topics import IMAGE_TOPICS, JOINT_TOPICS, TF_TOPIC, _spec


EEF_TOPICS = [
    "/dp/eef_actual",
    "/dp/eef_target",
]


def dp_topic_specs():
    """Return the RGB-only DP collection topic set.

    Depth images and camera calibration topics are deliberately excluded.
    Robot joint feedback/commands, TF, and dual-arm EEF vectors are retained.
    """
    specs = [_spec(topic, CompressedImage, "best_effort")
             for topic in IMAGE_TOPICS]
    specs += [_spec(topic, JointState, "reliable")
              for topic in JOINT_TOPICS]
    specs += [_spec(TF_TOPIC, TFMessage, "reliable")]
    specs += [_spec(topic, Float64MultiArray, "reliable")
              for topic in EEF_TOPICS]
    return specs
