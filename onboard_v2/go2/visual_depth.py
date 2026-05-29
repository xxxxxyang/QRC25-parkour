import argparse
import json
import os.path as osp
import time
from collections import OrderedDict

import numpy as np
import pyrealsense2 as rs
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray

import rclpy


@torch.no_grad()
def resize2d(img, resized_wh):
    resized_hw = (resized_wh[1], resized_wh[0])
    return TF.resize(img, resized_hw, interpolation=InterpolationMode.BICUBIC)


def image_msg_from_numpy(array, encoding):
    array = np.ascontiguousarray(array)
    msg = Image()
    msg.height = int(array.shape[0])
    msg.width = int(array.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(array.strides[0])
    msg.data = array.tobytes()
    return msg


class DepthPublisherNode(Node):
    def __init__(
        self,
        cfg,
        cropping=(0, 0, 0, 0),
        rs_resolution=(480, 270),
        rs_fps=30,
        depth_input_topic="/camera/forward_depth",
        rgb_topic="/camera/forward_rgb",
        camera_info_topic="/camera/camera_info",
        enable_rgb=False,
        forward_depth_topic="/forward_depth_image",
        debug=False,
    ):
        super().__init__("forward_depth_embedding")
        self.cfg = cfg
        self.cropping = cropping
        self.rs_resolution = rs_resolution
        self.rs_fps = rs_fps
        self.depth_input_topic = depth_input_topic
        self.rgb_topic = rgb_topic
        self.camera_info_topic = camera_info_topic
        self.enable_rgb = enable_rgb
        self.forward_depth_topic = forward_depth_topic
        self.debug = debug
        self._first_depth_published = False

        self.original_resolution = self.cfg["depth"].get("original", (106, 60))
        self.output_resolution = self.cfg["depth"].get("resized", (87, 58))
        self.near_clip = self.cfg["depth"].get("near_clip", 0.0)
        self.far_clip = self.cfg["depth"].get("far_clip", 2.0)
        self.depth_range = (self.near_clip * 1000, self.far_clip * 1000)

        self._start_pipeline()
        self._start_ros_handlers()

    def _start_pipeline(self):
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.depth,
            self.rs_resolution[0],
            self.rs_resolution[1],
            rs.format.z16,
            self.rs_fps,
        )
        if self.enable_rgb:
            self.rs_config.enable_stream(
                rs.stream.color,
                self.rs_resolution[0],
                self.rs_resolution[1],
                rs.format.rgb8,
                self.rs_fps,
            )
        self.rs_profile = self.rs_pipeline.start(self.rs_config)
        self.rs_align = rs.align(rs.stream.depth)
        self.rs_spatial_filter = rs.spatial_filter()
        self.rs_spatial_filter.set_option(rs.option.filter_magnitude, 5)
        self.rs_spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.rs_spatial_filter.set_option(rs.option.filter_smooth_delta, 1)
        self.rs_spatial_filter.set_option(rs.option.holes_fill, 4)
        self.rs_temporal_filter = rs.temporal_filter()
        self.rs_temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.rs_temporal_filter.set_option(rs.option.filter_smooth_delta, 1)
        self.rs_hole_filling_filter = rs.hole_filling_filter()
        self.get_logger().info(
            "RealSense depth pipeline started: "
            f"{self.rs_resolution[0]}x{self.rs_resolution[1]} @ {self.rs_fps}Hz"
        )

    def _start_ros_handlers(self):
        self.depth_input_pub = self.create_publisher(Image, self.depth_input_topic, 1)
        if self.enable_rgb:
            self.rgb_pub = self.create_publisher(Image, self.rgb_topic, 1)
            self.camera_info_pub = self.create_publisher(
                CameraInfo, self.camera_info_topic, 1
            )
        self.forward_depth_image_pub = self.create_publisher(
            Float32MultiArray, self.forward_depth_topic, 1
        )
        self.get_logger().info(
            "Depth publishers ready: "
            f"{self.forward_depth_topic} -> Float32MultiArray, "
            f"{self.depth_input_topic} -> Image"
        )

    def _get_frame(self):
        top, bottom, left, right = self.cropping
        h_end = None if bottom == 0 else -bottom
        w_end = None if right == 0 else -right
        latency_range = self.cfg["depth"].get("latency_range", [0.08, 0.142])
        rs_frame = self.rs_pipeline.wait_for_frames(int(latency_range[1] * 1000))
        if self.enable_rgb:
            rs_frame = self.rs_align.process(rs_frame)
        depth_frame = rs_frame.get_depth_frame()
        if not depth_frame:
            self.get_logger().error("No depth frame", throttle_duration_sec=1)
            return None
        if self.enable_rgb:
            color_frame = rs_frame.get_color_frame()
            if color_frame:
                rgb_image_np = np.asanyarray(color_frame.get_data())
                rgb_image_np = rgb_image_np[top:h_end, left:w_end]
                rgb_image_msg = image_msg_from_numpy(rgb_image_np, encoding="rgb8")
                rgb_image_msg.header.stamp = self.get_clock().now().to_msg()
                rgb_image_msg.header.frame_id = "d435_sim_depth_link"
                self.rgb_pub.publish(rgb_image_msg)

        depth_frame = self.rs_hole_filling_filter.process(depth_frame)
        depth_frame = self.rs_spatial_filter.process(depth_frame)
        depth_frame = self.rs_temporal_filter.process(depth_frame)

        depth_np = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
        depth = depth[:, :, top:h_end, left:w_end]
        depth = torch.clip(depth * 0.001, self.near_clip, self.far_clip)
        depth = resize2d(depth, self.output_resolution)
        depth = depth / (self.far_clip - self.near_clip) - 0.5

        depth_input_data = (
            (depth.detach().cpu().numpy() + 0.5) * (self.far_clip - self.near_clip)
            + self.near_clip
        )
        depth_input_data = (depth_input_data * 1000).astype(np.uint16)[0, 0]
        if self.debug:
            depth_input_data = depth_input_data.copy()
            depth_input_data[int(depth_input_data.shape[0] / 2), :] = 0
            depth_input_data[:, int(depth_input_data.shape[1] / 2)] = 0
        depth_input_msg = image_msg_from_numpy(depth_input_data, encoding="16UC1")
        depth_input_msg.header.stamp = self.get_clock().now().to_msg()
        depth_input_msg.header.frame_id = "d435_sim_depth_link"
        self.depth_input_pub.publish(depth_input_msg)

        return depth

    def publish_depth(self):
        depth = self._get_frame()
        if depth is None:
            return
        msg = Float32MultiArray()
        msg.data = depth.flatten().detach().cpu().numpy().tolist()
        self.forward_depth_image_pub.publish(msg)
        if not self._first_depth_published:
            self.get_logger().info(
                "First depth frame published: "
                f"shape={tuple(depth.shape)}, "
                f"range=({float(depth.min()):.4f}, {float(depth.max()):.4f})"
            )
            self._first_depth_published = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--enable-rgb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--crop-left", type=int, default=0)
    parser.add_argument("--crop-right", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    args = parser.parse_args()

    with open(osp.join(args.logdir, "config.json"), "r") as f:
        cfg = json.load(f, object_pairs_hook=OrderedDict)

    rclpy.init()
    node = DepthPublisherNode(
        cfg=cfg,
        cropping=(args.crop_top, args.crop_bottom, args.crop_left, args.crop_right),
        rs_resolution=(args.width, args.height),
        rs_fps=args.fps,
        enable_rgb=args.enable_rgb,
        debug=args.debug,
    )
    node.get_logger().info(
        "visual_depth is running; press Ctrl-C to stop. "
        f"publish_period={cfg['depth'].get('update_interval', 5) * cfg['control']['decimation'] * cfg['sim']['dt']:.3f}s"
    )

    try:
        while rclpy.ok():
            node.publish_depth()
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(cfg["depth"].get("update_interval", 5) * cfg["control"]["decimation"] * cfg["sim"]["dt"])
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
