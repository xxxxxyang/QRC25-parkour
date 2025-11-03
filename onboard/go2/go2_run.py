import rclpy
from rclpy.node import Node
from unitree_ros2_real import UnitreeRos2Real, get_euler_xyz
from go2_ros2_real import Go2Ros2Real, get_euler_xyz

import os
import os.path as osp
import json
import time
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rsl_rl import modules
from sport_api_constants import *

class Go2Node(Go2Ros2Real):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_class_name= "Go2", **kwargs)

        self.debug = bool(kwargs.get('debug', False))
        self.use_sport_policy = True
        self.use_parkour_policy = False
        self.global_counter = 0
        self.visual_update_interval = self.cfg["depth"].get("update_interval", 5)

    def warm_up(self):
        """
        switch to stand and warm up the robot, prepare for the parkour policy
        """
        # warm up the robot
        self.get_logger().info("Warming up the robot...")
        self.send_action(torch.zeros(self.num_actions, device= self.device))
        time.sleep(1.0) # wait for 1 sec
        self.get_logger().info("Robot is warmed up.")

    def register_models(self, task_model, task_policy):
        # self.stand_model = stand_model
        self.task_model = task_model

        self.task_policy = task_policy
        # self.use_stand_policy = True # Start with standing model

    def start_main_loop_timer(self, duration):
        self.main_loop_timer = self.create_timer(
            duration, # in sec
            self.main_loop,
        )
        
    def main_loop(self):
        if self.use_sport_mode:
            if (self.joy_stick_buffer.keys & self.WirelessButtons.R1):
                self.get_logger().info("In the sport mode, R1 pressed, robot will stand up.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDUP)
            if (self.joy_stick_buffer.keys & self.WirelessButtons.R2):
                self.get_logger().info("In the sport mode, R2 pressed, robot will sit down.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDDOWN)
            if (self.joy_stick_buffer.keys & self.WirelessButtons.X):
                self.get_logger().info("In the sport mode, X pressed, robot will balance stand.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)
            if (self.joy_stick_buffer.keys & self.WirelessButtons.L1):
                self.get_logger().info("Exist the sport mode. Switch to parkour policy.")
                self.use_sport_mode = False
                self._sport_state_change(0)
                self.use_parkour_policy = True
                # warm up the robot
                self.global_counter = 0
                self.warm_up()

        if self.use_parkour_policy:
            start_time = time.monotonic()
            # obs = self.get_obs()
            proprio = self.get_proprio()
            proprio_history = self._get_history_proprio()
            if self.global_counter % self.visual_update_interval == 0:
                depth_image = self._get_depth_image()
                if self.global_counter == 0:
                    self.last_depth_image = depth_image
                self.depth_latent_yaw = self.depth_encode(self.last_depth_image, proprio)
                self.last_depth_image = depth_image
            obs_time = time.monotonic()
            action = self.task_policy(proprio, proprio_history, self.depth_latent_yaw)
            policy_time = time.monotonic()
            self.send_action(action)
            publish_time = time.monotonic()
            if self.debug:
                print(
                    "obs_time: {:.5f}".format(obs_time - start_time),
                    "policy_time: {:.5f}".format(policy_time - obs_time),
                    "publish_time: {:.5f}".format(publish_time - policy_time),
                )
            if (self.joy_stick_buffer.keys & self.WirelessButtons.L2):
                self.get_logger().info("Exit the parkour policy. Switch to sport mode.")
                self.use_parkour_policy = False
                self.use_sport_mode = True
                self._sport_state_change(0)

        if (self.joy_stick_buffer.keys & self.WirelessButtons.Y):
            self.get_logger().info("Y pressed, reset the policy")
            self.task_model.reset()
            self.global_counter = 0

@torch.inference_mode()
def main(args):
    rclpy.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # config loading
    assert args.logdir is not None, "Please provide a logdir"
    with open(osp.join(args.logdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook= OrderedDict)
    # jit model path
    depth_actor_path = max(
        (osp.join(args.logdir, f) for f in os.listdir(args.logdir) if f.endswith("base_jit.pt")),
        key=lambda x: int(x.split('-')[-2]),  # select the latest checkpoint
        default=None
    )
    assert depth_actor_path is not None, "No depth_actor jit model found in the logdir"
    print("Loading depth_actor model from: ", depth_actor_path)
    depth_actor_model = torch.jit.load(depth_actor_path, map_location= device)
    depth_actor_model.eval()
    
    duration = config_dict["sim"]["dt"] * config_dict["control"]["decimation"] # in sec
    n_hist_len = config_dict["env"]["history_len"]
    n_proprio = config_dict["env"]["n_proprio"]

    env_node = Go2Node(
        "go2",
        cfg = config_dict,
        model_device = device,
        dryrun = not args.nodryrun,
        debug = args.debug,
    )

    env_node.get_logger().info("Model loaded from: {}".format(depth_actor_path))
    env_node.get_logger().info("Control Duration: {} sec".format(duration))
    env_node.get_logger().info("Motor Stiffness (kp): {}".format(env_node.p_gains))
    env_node.get_logger().info("Motor Damping (kd): {}".format(env_node.d_gains))


    # # zero_act_model to start the safe standing
    # zero_act_model = ZeroActModel()
    # zero_act_model = torch.jit.script(zero_act_model)

    # Construct task policy
    @torch.jit.script
    def policy(proprio: torch.Tensor, proprio_history: torch.Tensor, depth_latent_yaw: torch.Tensor):
        estimator = depth_actor_model.estimator.estimator
        hist_encoder = depth_actor_model.actor.history_encoder
        actor = depth_actor_model.actor.actor_backbone
        depth_latent = depth_latent_yaw[:, :-2]
        yaw = depth_latent_yaw[:, -2:] * 1.5
        proprio[:, 6:8] = yaw
        lin_vel_latent = estimator(proprio)
        activation = nn.ELU()
        priv_latent = hist_encoder(activation, proprio_history.view(-1, n_hist_len, n_proprio))
        obs = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)
        action = actor(obs)
        return action

    
    env_node.register_models(
        # zero_act_model,
        depth_actor_model,
        policy,
    )
    env_node.start_ros_handlers()
    if args.loop_mode == "while":
        rclpy.spin_once(env_node, timeout_sec= 0.)
        env_node.get_logger().info("Model and Policy are ready")
        while rclpy.ok():
            main_loop_time = time.monotonic()
            env_node.main_loop()
            rclpy.spin_once(env_node, timeout_sec= 0.)
            time.sleep(max(0, duration - (time.monotonic() - main_loop_time)))
    elif args.loop_mode == "timer":
        env_node.start_main_loop_timer(duration)
        rclpy.spin(env_node)

    rclpy.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--logdir", type= str, default= None, help= "The directory which contains the config.json and model_*.pt files")
    parser.add_argument("--nodryrun", action= "store_true", default= False, help= "Disable dryrun mode")
    parser.add_argument("--loop_mode", type= str, default= "timer",
        choices= ["while", "timer"],
        help= "Select which mode to run the main policy control iteration",
    )
    parser.add_argument("--debug", action= "store_true", default= False, help= "Enable debug mode for run node")

    args = parser.parse_args()
    main(args)
