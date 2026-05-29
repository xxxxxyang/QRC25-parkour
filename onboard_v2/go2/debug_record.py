#!/usr/bin/env python3

import argparse
import csv
import json
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from unitree_api.msg import Request
from unitree_go.msg import LowCmd, LowState, WirelessController


def now_label():
    return time.strftime("%Y%m%d_%H%M%S")


class CsvLog:
    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


class JsonlLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w")

    def write(self, obj):
        self.file.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class ParkourDebugRecorder(Node):
    def __init__(self, output_dir, lowcmd_topic=None, lowstate_hz=10.0):
        super().__init__("parkour_debug_recorder")
        self.output_dir = Path(output_dir)
        self.lowcmd_topic = lowcmd_topic
        self.lowcmd_sub = None
        self.lowstate_period = 1.0 / lowstate_hz if lowstate_hz > 0 else 0.0
        self.last_lowstate_t = 0.0

        self.wireless_log = CsvLog(
            self.output_dir / "wireless.csv",
            ["t", "keys", "lx", "ly", "rx", "ry"],
        )
        lowstate_fields = ["t", "tick", "gyro_x", "gyro_y", "gyro_z", "rpy_r", "rpy_p", "rpy_y"]
        lowstate_fields += [f"q{i}" for i in range(12)]
        lowstate_fields += [f"dq{i}" for i in range(12)]
        lowstate_fields += [f"foot{i}" for i in range(4)]
        self.lowstate_log = CsvLog(self.output_dir / "lowstate.csv", lowstate_fields)

        lowcmd_fields = ["t", "topic", "crc"]
        lowcmd_fields += [f"mode{i}" for i in range(12)]
        lowcmd_fields += [f"q{i}" for i in range(12)]
        lowcmd_fields += [f"kp{i}" for i in range(12)]
        lowcmd_fields += [f"kd{i}" for i in range(12)]
        self.lowcmd_log = CsvLog(self.output_dir / "lowcmd.csv", lowcmd_fields)

        self.depth_log = CsvLog(
            self.output_dir / "depth_stats.csv",
            ["t", "topic", "length", "min", "max", "mean"],
        )
        self.image_log = CsvLog(
            self.output_dir / "image_stats.csv",
            ["t", "topic", "height", "width", "encoding", "step", "data_len"],
        )
        self.request_log = JsonlLog(self.output_dir / "requests.jsonl")

        self.create_subscription(WirelessController, "/wirelesscontroller", self.on_wireless, 10)
        self.create_subscription(LowState, "/lowstate", self.on_lowstate, 10)
        self.create_subscription(Float32MultiArray, "/forward_depth_image", self.on_depth, 10)
        self.create_subscription(Image, "/camera/forward_depth", self.on_image, 10)
        self.create_subscription(Request, "/api/sport/request", self.on_sport_request, 10)
        self.create_subscription(Request, "/api/robot_state/request", self.on_robot_state_request, 10)
        self.scan_timer = self.create_timer(1.0, self.scan_lowcmd_topic)

        self.get_logger().info(f"Writing debug logs to {self.output_dir}")
        if self.lowcmd_topic:
            self.attach_lowcmd_topic(self.lowcmd_topic)
        else:
            self.get_logger().info("Waiting for /lowcmd_dryrun_* topic.")

    def stamp(self):
        return time.time()

    def attach_lowcmd_topic(self, topic):
        if self.lowcmd_sub is not None:
            return
        self.lowcmd_topic = topic
        self.lowcmd_sub = self.create_subscription(
            LowCmd,
            topic,
            lambda msg: self.on_lowcmd(topic, msg),
            10,
        )
        self.get_logger().info(f"Recording lowcmd topic: {topic}")

    def scan_lowcmd_topic(self):
        if self.lowcmd_sub is not None:
            return
        for topic, types in self.get_topic_names_and_types():
            if topic.startswith("/lowcmd_dryrun_"):
                self.attach_lowcmd_topic(topic)
                return

    def on_wireless(self, msg):
        self.wireless_log.write(
            {
                "t": self.stamp(),
                "keys": msg.keys,
                "lx": msg.lx,
                "ly": msg.ly,
                "rx": msg.rx,
                "ry": msg.ry,
            }
        )

    def on_lowstate(self, msg):
        t = self.stamp()
        if self.lowstate_period > 0 and t - self.last_lowstate_t < self.lowstate_period:
            return
        self.last_lowstate_t = t
        row = {
            "t": t,
            "tick": msg.tick,
            "gyro_x": msg.imu_state.gyroscope[0],
            "gyro_y": msg.imu_state.gyroscope[1],
            "gyro_z": msg.imu_state.gyroscope[2],
            "rpy_r": msg.imu_state.rpy[0],
            "rpy_p": msg.imu_state.rpy[1],
            "rpy_y": msg.imu_state.rpy[2],
        }
        for i in range(12):
            row[f"q{i}"] = msg.motor_state[i].q
            row[f"dq{i}"] = msg.motor_state[i].dq
        for i in range(4):
            row[f"foot{i}"] = msg.foot_force[i]
        self.lowstate_log.write(row)

    def on_lowcmd(self, topic, msg):
        row = {"t": self.stamp(), "topic": topic, "crc": msg.crc}
        for i in range(12):
            row[f"mode{i}"] = msg.motor_cmd[i].mode
            row[f"q{i}"] = msg.motor_cmd[i].q
            row[f"kp{i}"] = msg.motor_cmd[i].kp
            row[f"kd{i}"] = msg.motor_cmd[i].kd
        self.lowcmd_log.write(row)

    def on_depth(self, msg):
        if not msg.data:
            self.depth_log.write(
                {"t": self.stamp(), "topic": "/forward_depth_image", "length": 0, "min": "", "max": "", "mean": ""}
            )
            return
        values = list(msg.data)
        self.depth_log.write(
            {
                "t": self.stamp(),
                "topic": "/forward_depth_image",
                "length": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
        )

    def on_image(self, msg):
        self.image_log.write(
            {
                "t": self.stamp(),
                "topic": "/camera/forward_depth",
                "height": msg.height,
                "width": msg.width,
                "encoding": msg.encoding,
                "step": msg.step,
                "data_len": len(msg.data),
            }
        )

    def on_sport_request(self, msg):
        self.request_log.write(
            {
                "t": self.stamp(),
                "topic": "/api/sport/request",
                "api_id": msg.header.identity.api_id,
                "parameter": msg.parameter,
            }
        )

    def on_robot_state_request(self, msg):
        self.request_log.write(
            {
                "t": self.stamp(),
                "topic": "/api/robot_state/request",
                "api_id": msg.header.identity.api_id,
                "parameter": msg.parameter,
            }
        )

    def close(self):
        for log in [
            self.wireless_log,
            self.lowstate_log,
            self.lowcmd_log,
            self.depth_log,
            self.image_log,
            self.request_log,
        ]:
            log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV/JSONL logs. Default: /tmp/parkour_debug_<timestamp>",
    )
    parser.add_argument(
        "--lowcmd-topic",
        default=None,
        help="Dryrun lowcmd topic. If omitted, auto-detect /lowcmd_dryrun_*.",
    )
    parser.add_argument("--lowstate-hz", type=float, default=10.0)
    args = parser.parse_args()

    output_dir = args.output_dir or f"/tmp/parkour_debug_{now_label()}"
    rclpy.init()
    node = ParkourDebugRecorder(output_dir, args.lowcmd_topic, args.lowstate_hz)
    try:
        rclpy.spin(node)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()
        print(f"Debug logs written to: {output_dir}")


if __name__ == "__main__":
    main()
