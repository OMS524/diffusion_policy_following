"""Thin wrapper around rosbag2_py.SequentialWriter for one episode.

Produces the same on-disk layout as the existing dataset:
    <bag_dir>/<basename>_0.db3   (sqlite3 storage, cdr serialization)
    <bag_dir>/metadata.yaml      (written on close, rosbag2 format v5+)
"""

import threading

import rosbag2_py
from rclpy.serialization import serialize_message


class EpisodeBagWriter:
    def __init__(self, bag_dir: str, topic_specs, storage_id: str = "sqlite3"):
        self._lock = threading.Lock()
        self._writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        self._writer.open(storage_options, converter_options)
        for spec in topic_specs:
            self._writer.create_topic(
                rosbag2_py.TopicMetadata(
                    name=spec.name,
                    type=spec.type_str,
                    serialization_format="cdr",
                )
            )
        self.counts = {spec.name: 0 for spec in topic_specs}
        self._closed = False

    def write(self, topic_name: str, msg, stamp_ns: int):
        """Serialize and append one message. Thread-safe (rosbag2 writer is not)."""
        data = serialize_message(msg)
        with self._lock:
            if self._closed:
                return
            self._writer.write(topic_name, data, stamp_ns)
            self.counts[topic_name] = self.counts.get(topic_name, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def close(self):
        """Flush metadata.yaml and release the writer."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # rosbag2_py finalizes metadata.yaml when the writer is destroyed.
            del self._writer
            self._writer = None
