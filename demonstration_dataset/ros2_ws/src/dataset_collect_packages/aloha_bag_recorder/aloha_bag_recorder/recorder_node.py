"""Episode-wise rosbag2 recorder node for the airo / Piper ALOHA setup.

The node subscribes to the canonical 9-topic set (see ``topics.py``) and writes
every received message into a per-episode ``.db3`` bag via ``EpisodeBagWriter``.
Recording is driven by three ``std_srvs/Trigger`` services so it scriptable from
the CLI or a teleop launch file:

    ros2 service call /aloha_recorder/start_recording  std_srvs/srv/Trigger
    ros2 service call /aloha_recorder/stop_recording   std_srvs/srv/Trigger
    ros2 service call /aloha_recorder/discard_recording std_srvs/srv/Trigger

Each ``start`` opens ``<save_dir>/[<task_name>/]episode_<unix_sec>``; ``stop``
finalizes it (and logs per-topic message counts), ``discard`` finalizes then
deletes the directory.

Subscriptions are created lazily on the first ``start_recording`` so their QoS
can be auto-negotiated from the *live* publishers (``resolve_qos``); this avoids
the silent "subscribed but no messages" failure that happens when a best_effort
sensor publisher meets a reliable subscriber.
"""

import os
import shutil
from functools import partial

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy
from std_srvs.srv import Trigger

from .topics import default_topic_specs, fallback_qos
from .bag_writer import EpisodeBagWriter


def resolve_qos(node: Node, spec, depth: int) -> QoSProfile:
    """Pick a subscription QoS compatible with whatever is currently publishing.

    Adopts the first live publisher's reliability + durability so a best_effort
    sensor stream is actually received, while forcing KEEP_LAST/``depth`` so the
    recorder never blocks an unbounded queue. Falls back to the spec default
    when no publisher is up yet.
    """
    infos = node.get_publishers_info_by_topic(spec.name)
    if not infos:
        node.get_logger().warn(
            f"no publisher on {spec.name} yet; using {spec.reliability} fallback QoS"
        )
        return fallback_qos(spec.reliability, depth)

    pub_qos = infos[0].qos_profile
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=pub_qos.reliability,
        durability=pub_qos.durability,
    )


class RecorderNode(Node):
    def __init__(self):
        super().__init__("aloha_recorder")

        self.declare_parameter("save_dir", "./raw_data")
        self.declare_parameter("task_name", "")
        self.declare_parameter("queue_depth", 100)
        # Auto-stop the episode this many seconds after start_recording. <= 0
        # disables it (manual stop_recording only), preserving the old behaviour.
        # When record_frames > 0 this acts only as a safety fallback timeout.
        self.declare_parameter("record_duration", 0.0)
        # Auto-stop after exactly this many frames of `frame_count_topic` (the
        # convert reference stream). > 0 takes precedence over record_duration and
        # yields exactly-equal episode lengths. <= 0 disables frame-count mode.
        self.declare_parameter("record_frames", 0)
        self.declare_parameter("frame_count_topic", "/cam_top/image_compressed")

        self._specs = default_topic_specs()
        self._subs = []                 # created lazily on first start
        self._writer = None             # active EpisodeBagWriter, or None
        self._episode_dir = None
        self._auto_timer = None         # one-shot auto-stop timer, or None
        self._record_duration = 0.0     # seconds; refreshed on each start
        self._target_frames = 0         # frame-count auto-stop target (0 = off)
        self._frame_count = 0           # frames seen on the count topic this episode
        self._count_topic = ""          # topic whose frames are counted

        self.create_service(Trigger, "~/start_recording", self._on_start)
        self.create_service(Trigger, "~/stop_recording", self._on_stop)
        self.create_service(Trigger, "~/discard_recording", self._on_discard)

        self.get_logger().info(
            "aloha_recorder ready. Call ~/start_recording to begin an episode."
        )

    # ------------------------------------------------------------------ subs
    def _ensure_subscriptions(self):
        """Create the generic subscriptions once, resolving QoS from live pubs.

        QoS is negotiated from the live publisher *at this moment* (first
        start_recording) and then frozen for the session. Any topic whose
        publisher is not up yet falls back to its default QoS, which for a
        best_effort sensor stream (depth/colour) can mean "subscribed but
        receives nothing". We therefore emit a single, loud warning listing the
        topics that had no publisher so the operator can bring the camera nodes
        up *before* the first start_recording.
        """
        if self._subs:
            return
        depth = self.get_parameter("queue_depth").value
        missing = []
        for spec in self._specs:
            if not self.get_publishers_info_by_topic(spec.name):
                missing.append(spec.name)
            qos = resolve_qos(self, spec, depth)
            sub = self.create_subscription(
                spec.msg_class,
                spec.name,
                partial(self._on_msg, spec.name),
                qos,
            )
            self._subs.append(sub)
        self.get_logger().info(f"subscribed to {len(self._subs)} topics")
        if missing:
            self.get_logger().warn(
                "QoS negotiated with NO live publisher for "
                f"{len(missing)} topic(s): {', '.join(missing)}. "
                "These were frozen to fallback QoS and may record ZERO messages. "
                "Bring all camera (incl. depth) and arm publishers up BEFORE the "
                "first start_recording, then restart the recorder if needed."
            )

    def _on_msg(self, topic_name: str, msg):
        writer = self._writer
        if writer is None:
            return
        stamp_ns = self.get_clock().now().nanoseconds
        writer.write(topic_name, msg, stamp_ns)

        # Frame-count auto-stop: finalize once the reference stream reaches the
        # target, so every episode lands on exactly the same number of base frames.
        if self._target_frames > 0 and topic_name == self._count_topic:
            self._frame_count += 1
            if self._frame_count >= self._target_frames:
                self._cancel_auto_timer()
                episode_dir, total = self._finalize_recording()
                self.get_logger().info(
                    f"auto-stopped at {self._frame_count} {self._count_topic} frames "
                    f"-> {episode_dir} ({total} msgs)")

    # -------------------------------------------------------------- services
    def _new_episode_dir(self) -> str:
        save_dir = os.path.expanduser(self.get_parameter("save_dir").value)
        task_name = self.get_parameter("task_name").value
        if task_name:
            save_dir = os.path.join(save_dir, task_name)
        sec = self.get_clock().now().seconds_nanoseconds()[0]
        base = os.path.join(save_dir, f"episode_{sec}")
        path = base
        suffix = 1
        while os.path.exists(path):
            path = f"{base}_{suffix}"
            suffix += 1
        return path

    def _on_start(self, request, response):
        if self._writer is not None:
            response.success = False
            response.message = f"already recording {self._episode_dir}"
            return response

        self._ensure_subscriptions()
        self._episode_dir = self._new_episode_dir()
        os.makedirs(os.path.dirname(self._episode_dir), exist_ok=True)
        try:
            self._writer = EpisodeBagWriter(self._episode_dir, self._specs)
        except Exception as exc:  # noqa: BLE001 - report any rosbag2 failure
            self._writer = None
            response.success = False
            response.message = f"failed to open bag: {exc}"
            return response

        self.get_logger().info(f"recording -> {self._episode_dir}")
        # Auto-stop mode selection. record_frames (exact length) takes precedence
        # over record_duration; both <= 0 means manual stop_recording only.
        self._target_frames = int(self.get_parameter("record_frames").value)
        self._frame_count = 0
        self._count_topic = str(self.get_parameter("frame_count_topic").value)
        self._record_duration = float(self.get_parameter("record_duration").value)
        if self._target_frames > 0:
            self.get_logger().info(
                f"auto-stop after {self._target_frames} frames of {self._count_topic}")
        if self._record_duration > 0.0:
            self._auto_timer = self.create_timer(self._record_duration, self._auto_stop)
            kind = "safety timeout" if self._target_frames > 0 else "auto-stop"
            self.get_logger().info(f"{kind} scheduled in {self._record_duration:.2f}s")
        response.success = True
        response.message = self._episode_dir
        return response

    def _cancel_auto_timer(self):
        if self._auto_timer is not None:
            self._auto_timer.cancel()
            self.destroy_timer(self._auto_timer)
            self._auto_timer = None

    def _finalize_recording(self):
        """Close the active writer, log per-topic counts, return (dir, total)."""
        writer, episode_dir = self._writer, self._episode_dir
        self._writer = None
        self._episode_dir = None
        self._target_frames = 0   # stop counting in any already-queued _on_msg calls
        counts = dict(writer.counts)
        total = writer.total
        writer.close()
        summary = ", ".join(f"{k}:{v}" for k, v in counts.items())
        self.get_logger().info(f"saved {episode_dir} ({total} msgs) [{summary}]")
        return episode_dir, total

    def _auto_stop(self):
        """Timer callback: finalize the episode after record_duration seconds."""
        self._cancel_auto_timer()
        if self._writer is None:
            return
        episode_dir, total = self._finalize_recording()
        self.get_logger().info(
            f"auto-stopped after {self._record_duration:.2f}s -> {episode_dir} ({total} msgs)")

    def _on_stop(self, request, response):
        if self._writer is None:
            response.success = False
            response.message = "not recording"
            return response

        self._cancel_auto_timer()
        episode_dir, total = self._finalize_recording()
        response.success = True
        response.message = f"{episode_dir} ({total} msgs)"
        return response

    def _on_discard(self, request, response):
        if self._writer is None:
            response.success = False
            response.message = "not recording"
            return response

        self._cancel_auto_timer()
        writer, episode_dir = self._writer, self._episode_dir
        self._writer = None
        self._episode_dir = None
        writer.close()
        if episode_dir and os.path.isdir(episode_dir):
            shutil.rmtree(episode_dir, ignore_errors=True)

        self.get_logger().info(f"discarded {episode_dir}")
        response.success = True
        response.message = f"discarded {episode_dir}"
        return response

    def destroy_node(self):
        # finalize a dangling episode so we never leave a half-written bag
        self._cancel_auto_timer()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
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
