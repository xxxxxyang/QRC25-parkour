import rclpy
from rclpy.node import Node
from unitree_ros2_real import UnitreeRos2Real
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CameraInfo

import os
import os.path as osp
import json
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from rsl_rl import modules

import pyrealsense2 as rs
import ros2_numpy as rnp

@torch.no_grad()
def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img), size)).data

class VisualHandlerNode(Node):
    """ A wapper class for the realsense camera """
    def __init__(self,
            cfg: dict,
            cropping: list = [0, 0, 0, 0], # top, bottom, left, right
            rs_resolution: tuple = (480, 270), # width, height for the realsense camera
            rs_fps: int = 30,
            depth_input_topic = "/camera/forward_depth",
            rgb_topic = "/camera/forward_rgb",
            camera_info_topic = "/camera/camera_info",
            enable_rgb = False,
            forward_depth_topic = "/forward_depth_image",
            debug = False
        ):
        super().__init__("forward_depth_embedding")
        self.cfg = cfg
        self.cropping = cropping
        self.rs_resolution = rs_resolution
        self.rs_fps = rs_fps
        self.depth_input_topic = depth_input_topic
        self.rgb_topic= rgb_topic
        self.camera_info_topic = camera_info_topic
        self.enable_rgb= enable_rgb
        self.forward_depth_topic = forward_depth_topic
        self.debug = debug

        self.parse_args()
        self.start_pipeline()
        self.start_ros_handlers()

    def parse_args(self):
        orig = self.cfg["depth"].get(
            "original",
            [87, 58]
        )   # (W, H)
        self.output_resolution = [orig[1], orig[0]] # (H, W)
        near = self.cfg["depth"].get("near_plane", 0.1) # [m]
        far = self.cfg["depth"].get("far_clip", 2.0) # [m]
        self.depth_range = (near * 1000, far * 1000) # [m] -> [mm]

    def start_pipeline(self):
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
        self.rs_profile = self.rs_pipeline.start(self.rs_config) # config profile
        if self.debug:
            info = VisualHandlerNode.get_rs_profile_info(self.rs_profile)
            # print out stream resolutions
            for name, s in info.get("streams", {}).items():
                w = s.get("width")
                h = s.get("height")
                if w is not None and h is not None:
                    self.get_logger().info(f"{name} resolution: {w}x{h}", once= True)
            # print out depth scale
            depth_scale = info.get("depth_scale", None)
            if depth_scale is not None:
                self.get_logger().info(f"Depth scale: {depth_scale} m/unit", once= True)

        self.rs_align = rs.align(rs.stream.depth)

        # build rs builtin filters
        # self.rs_decimation_filter = rs.decimation_filter()
        # self.rs_decimation_filter.set_option(rs.option.filter_magnitude, 6)
        self.rs_hole_filling_filter = rs.hole_filling_filter()
        self.rs_spatial_filter = rs.spatial_filter()
        self.rs_spatial_filter.set_option(rs.option.filter_magnitude, 5)
        self.rs_spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.rs_spatial_filter.set_option(rs.option.filter_smooth_delta, 1)
        self.rs_spatial_filter.set_option(rs.option.holes_fill, 4)
        self.rs_temporal_filter = rs.temporal_filter()
        self.rs_temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.rs_temporal_filter.set_option(rs.option.filter_smooth_delta, 1)
        # using a list of filters to define the filtering order
        self.rs_filters = [
            # self.rs_decimation_filter,
            self.rs_hole_filling_filter,
            self.rs_spatial_filter,
            self.rs_temporal_filter,
        ]

        if self.enable_rgb:
            # get frame with longer waiting time to start the system
            # I know what's going on, but when enabling rgb, this solves the problem.
            latency_range = self.cfg["sensor"]["forward_camera"]["latency_range"] if self.cfg["sensor"]["forward_camera"].get("latency_range", None) is not None else [0.08, 0.142]
            rs_frame = self.rs_pipeline.wait_for_frames(int(
                latency_range[1] * 10000 # ms * 10
            ))

    def start_ros_handlers(self):
        self.depth_input_pub = self.create_publisher(
            Image,
            self.depth_input_topic,
            1,
        )
        if self.enable_rgb:
            self.rgb_pub = self.create_publisher(
                Image,
                self.rgb_topic,
                1,
            )
            self.camera_info_pub = self.create_publisher(
                CameraInfo,
                self.camera_info_topic,
                1,
            )
            # fill in critical info of processed camera info based on simulated data
            # NOTE: simply because realsense's camera_info does not match our network input.
            # It is easier to compute this way.
            self.camera_info_msg = CameraInfo()
            self.camera_info_msg.header.frame_id = "d435_sim_depth_link"
            self.camera_info_msg.height = self.output_resolution[0]
            self.camera_info_msg.width = self.output_resolution[1]
            self.camera_info_msg.distortion_model = "plumb_bob"
            self.camera_info_msg.d = [0., 0., 0., 0., 0.]
            sim_raw_resolution = self.cfg["sensor"]["forward_camera"]["resolution"]
            sim_cropping_h = self.cfg["sensor"]["forward_camera"]["crop_top_bottom"]
            sim_cropping_w = self.cfg["sensor"]["forward_camera"]["crop_left_right"]
            cropped_resolution = [ # (H, W)
                sim_raw_resolution[0] - sum(sim_cropping_h),
                sim_raw_resolution[1] - sum(sim_cropping_w),
            ]
            network_input_resolution = self.cfg["sensor"]["forward_camera"]["output_resolution"]
            x_fov = sum(self.cfg["sensor"]["forward_camera"]["horizontal_fov"]) / 2 / 180 * np.pi
            fx = (sim_raw_resolution[1]) / (2 * np.tan(x_fov / 2))
            fy = fx
            fx = fx * network_input_resolution[1] / cropped_resolution[1]
            fy = fy * network_input_resolution[0] / cropped_resolution[0]
            cx = (sim_raw_resolution[1] / 2) - sim_cropping_w[0]
            cy = (sim_raw_resolution[0] / 2) - sim_cropping_h[0]
            cx = cx * network_input_resolution[1] / cropped_resolution[1]
            cy = cy * network_input_resolution[0] / cropped_resolution[0]
            self.camera_info_msg.k = [
                fx, 0., cx,
                0., fy, cy,
                0., 0., 1.,
            ]
            self.camera_info_msg.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
            self.camera_info_msg.p = [
                fx, 0., cx, 0.,
                0., fy, cy, 0.,
                0., 0., 1., 0.,
            ]
            self.camera_info_msg.binning_x = 0
            self.camera_info_msg.binning_y = 0
            self.camera_info_msg.roi.do_rectify = False
            self.create_timer(
                self.cfg["sensor"]["forward_camera"]["refresh_duration"],
                self.publish_camera_info_callback,
            )

        self.forward_depth_image_pub = self.create_publisher(
            Float32MultiArray,
            self.forward_depth_topic,
            1,
        )
        self.get_logger().info("ros handlers started")

    def publish_camera_info_callback(self):
        self.camera_info_msg.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info("camera info published", once= True)
        self.camera_info_pub.publish(self.camera_info_msg)

    def get_depth_frame(self):
        # read from pyrealsense2, preprocess and write the model embedding to the buffer
        top, bottom, left, right = self.cropping
        h_end = None if bottom == 0 else -bottom
        w_end = None if right == 0 else -right
        latency_range = self.cfg["depth"]["latency_range"] if self.cfg["depth"].get("latency_range", None) is not None else [0.08, 0.142]
        rs_frame = self.rs_pipeline.wait_for_frames(int(
            latency_range[1] * 1000 # ms
        ))
        if self.enable_rgb:
            rs_frame = self.rs_align.process(rs_frame)
        depth_frame = rs_frame.get_depth_frame()
        if not depth_frame:
            self.get_logger().error("No depth frame", throttle_duration_sec= 1)
            return
        color_frame = rs_frame.get_color_frame()
        if color_frame:
            rgb_image_np = np.asanyarray(color_frame.get_data())
            # rgb_image_np = np.rot90(rgb_image_np, k= 2) # since the camera is inverted
            rgb_image_np = rgb_image_np[
                top:h_end,
                left:w_end,
            ]
            rgb_image_msg = rnp.msgify(Image, rgb_image_np, encoding= "rgb8")
            rgb_image_msg.header.stamp = self.get_clock().now().to_msg()
            rgb_image_msg.header.frame_id = "d435_sim_depth_link"
            self.rgb_pub.publish(rgb_image_msg)
            self.get_logger().info("rgb image published", once= True)
        
        # apply relsense filters
        for rs_filter in self.rs_filters:
            depth_frame = rs_filter.process(depth_frame)
        # TODO: check if depth_frame shape is (1, 1, H, W)
        depth_image_np = np.asanyarray(depth_frame.get_data())
        # depth_image_np = np.rot90(depth_image_np, k= 2) # k = 2 for rotate 90 degree twice
        depth_image_pyt = torch.from_numpy(depth_image_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        
        # apply torch filters
        depth_image_pyt = depth_image_pyt[
            :, :,
            top:h_end,
            left:w_end,
        ]
        depth_image_pyt = torch.clip(depth_image_pyt, self.depth_range[0], self.depth_range[1]) / (self.depth_range[1] - self.depth_range[0])
        depth_image_pyt = resize2d(depth_image_pyt, self.output_resolution)
        depth_image_pyt -= 0.5 # normalize to [-0.5, 0.5]

        # publish the depth image input to ros topic
        self.get_logger().info("depth range: {}-{}".format(*self.depth_range), once= True)
        depth_input_data = (
            (depth_image_pyt.detach().cpu().numpy()+0.5) * (self.depth_range[1] - self.depth_range[0]) + self.depth_range[0]
        ).astype(np.uint16)[0, 0] # (h, w) unit [mm]
        # DEBUG: centering the depth image
        if self.debug:
            depth_input_data = depth_input_data.copy()
            depth_input_data[int(depth_input_data.shape[0] / 2), :] = 0
            depth_input_data[:, int(depth_input_data.shape[1] / 2)] = 0

        depth_input_msg = rnp.msgify(Image, depth_input_data, encoding= "16UC1")
        depth_input_msg.header.stamp = self.get_clock().now().to_msg()
        depth_input_msg.header.frame_id = "d435_sim_depth_link"
        self.depth_input_pub.publish(depth_input_msg)
        self.get_logger().info("depth input published", once= True)

        return depth_image_pyt
    
    def get_rs_profile_info(profile):
        info = {}
        dev = profile.get_device()
        info['serial_number'] = dev.get_info(rs.camera_info.serial_number)
        info['product_line'] = dev.get_info(rs.camera_info.product_line)

        # depth scale (if available)
        try:
            depth_sensor = dev.first_depth_sensor()
            info['depth_scale'] = depth_sensor.get_depth_scale()
        except Exception:
            info['depth_scale'] = None

        streams = {}
        for sp in profile.get_streams():
            if sp.stream_type() == rs.stream.depth:
                name = 'depth'
            elif sp.stream_type() == rs.stream.color:
                name = 'color'
            elif sp.stream_type() == rs.stream.infrared:
                name = 'infrared'
            else:
                name = str(sp.stream_type())

            s = {'format': str(sp.format()), 'fps': sp.fps()}
            try:
                vsp = sp.as_video_stream_profile()
                intr = vsp.get_intrinsics()
                s.update({
                    'width': vsp.width,
                    'height': vsp.height,
                    'intrinsics': {
                        'fx': intr.fx,
                        'fy': intr.fy,
                        'ppx': intr.ppx,
                        'ppy': intr.ppy,
                        'model': int(intr.model),
                        'coeffs': list(intr.coeffs),
                    }
                })
            except Exception:
                # not a video profile or intrinsics not available
                pass

            streams[name] = s

        info['streams'] = streams

        # extrinsics between depth and color (if both present)
        try:
            depth_vsp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            color_vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
            extr = depth_vsp.get_extrinsics_to(color_vsp)
            info['extrinsics_depth_to_color'] = {
                'rotation': list(extr.rotation),
                'translation': list(extr.translation),
            }
        except Exception:
            pass

        return info
    
    def publish_depth_data(self, depth_data):
        msg = Float32MultiArray()
        msg.data = depth_data.flatten().detach().cpu().numpy().tolist()
        self.forward_depth_image_pub.publish(msg)
        self.get_logger().info("depth data published", once= True)

    def start_main_loop_timer(self, duration):
        self.create_timer(
            duration,
            self.main_loop,
        )

    def main_loop(self):
        depth_image_pyt = self.get_depth_frame()
        if depth_image_pyt is not None:
            self.publish_depth_data(depth_image_pyt)
        else:
            self.get_logger().warn("One frame of depth embedding if not acquired")

@torch.inference_mode()
def main(args):
    rclpy.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert args.logdir is not None, "Please provide a logdir"
    with open(osp.join(args.logdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook= OrderedDict)
        
    duration = 0.01 # in sec

    visual_node = VisualHandlerNode(
        cfg= config_dict,
        cropping= [args.crop_top, args.crop_bottom, args.crop_left, args.crop_right],
        rs_resolution= (args.width, args.height),
        rs_fps= args.fps,
        enable_rgb= args.rgb,
        debug= args.debug,
    )

    if args.loop_mode == "while":
        rclpy.spin_once(visual_node, timeout_sec= 0.)
        while rclpy.ok():
            main_loop_time = time.monotonic()
            visual_node.main_loop()
            rclpy.spin_once(visual_node, timeout_sec= 0.)
            time.sleep(max(0, duration - (time.monotonic() - main_loop_time)))
    elif args.loop_mode == "timer":
        visual_node.start_main_loop_timer(duration)
        rclpy.spin(visual_node)

    visual_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir", type= str, default= None, help= "The directory which contains the config.json and model_*.pt files")
    
    parser.add_argument("--height",
        type= int,
        default= 480,
        help= "The height of the realsense image",
    )
    parser.add_argument("--width",
        type= int,
        default= 640,
        help= "The width of the realsense image",
    )
    parser.add_argument("--fps",
        type= int,
        default= 30,
        help= "The fps request to the rs pipeline",
    )
    parser.add_argument("--crop_left",
        type= int,
        default= 28,
        help= "num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--crop_right",
        type= int,
        default= 36,
        help= "num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--crop_top",
        type= int,
        default= 48,
        help= "num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--crop_bottom",
        type= int,
        default= 0,
        help= "num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--rgb",
        action= "store_true",
        default= False,
        help= "Set to enable rgb visualization",
    )
    parser.add_argument("--loop_mode", type= str, default= "timer",
        choices= ["while", "timer"],
        help= "Select which mode to run the main policy control iteration",
    )
    parser.add_argument("--debug", 
        action= "store_true",
        default= False,
        help= "Set to enable debug mode",
    )

    args = parser.parse_args()
    main(args)
