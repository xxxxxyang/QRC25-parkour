import argparse
import json
import os.path as osp
import logging
import time
from collections import OrderedDict
from pathlib import Path

import rclpy
import torch
from std_msgs.msg import Float32MultiArray

try:
    from .go2_ros2_real import Go2Ros2Real
    from .sport_api_constants import (
        ROBOT_SPORT_API_ID_BALANCESTAND,
        ROBOT_SPORT_API_ID_STANDDOWN,
        ROBOT_SPORT_API_ID_STANDUP,
    )
    from .vision_policy import load_hardware_vision_policy
except ImportError:
    from go2_ros2_real import Go2Ros2Real
    from sport_api_constants import (
        ROBOT_SPORT_API_ID_BALANCESTAND,
        ROBOT_SPORT_API_ID_STANDDOWN,
        ROBOT_SPORT_API_ID_STANDUP,
    )
    from vision_policy import load_hardware_vision_policy


class Go2DepthPolicyNode(Go2Ros2Real):
    def __init__(self, *args, **kwargs):
        self.debug = bool(kwargs.pop("debug", False))
        self.debug_logger = kwargs.pop("debug_logger", None)
        kwargs.setdefault("depth_data_topic", None)
        super().__init__(*args, robot_class_name="Go2", **kwargs)
        self.use_sport_mode = True
        self.use_parkour_policy = False
        self.global_counter = 0
        self.visual_update_interval = self.cfg["depth"].get("update_interval", 5)
        self.depth_image_buffer = None
        self.policy_model = None
        self._first_depth_received = False
        self._first_action_sent = False
        self._last_keys = None

        self.depth_sub = self.create_subscription(
            Float32MultiArray,
            "/forward_depth_image",
            self._depth_callback,
            1,
        )

    def _depth_callback(self, msg):
        resolution = self.cfg["depth"].get("resized", [87, 58])
        depth = torch.tensor(msg.data, dtype=torch.float32, device=self.model_device)
        self.depth_image_buffer = depth.view(1, resolution[1], resolution[0])
        if not self._first_depth_received:
            self.get_logger().info(
                "First depth frame received: "
                f"shape={tuple(self.depth_image_buffer.shape)}, "
                f"finite={bool(torch.isfinite(self.depth_image_buffer).all())}"
            )
            self._first_depth_received = True

    def attach_policy(self, policy_model):
        self.policy_model = policy_model

    def attach_debug_logger(self, debug_logger):
        self.debug_logger = debug_logger

    def _debug_log(self, message):
        if self.debug and self.debug_logger is not None:
            self.debug_logger.info(message)

    def reset_policy_state(self):
        if self.policy_model is not None:
            self.policy_model.reset()
        self.global_counter = 0

    def start_main_loop_timer(self, duration):
        self.main_loop_timer = self.create_timer(duration, self.main_loop)

    def main_loop(self):
        keys = self.joy_stick_buffer.keys
        if self._last_keys != keys:
            self.get_logger().info(
                "Wireless keys changed: "
                f"keys={keys}, lx={self.joy_stick_buffer.lx:.3f}, "
                f"ly={self.joy_stick_buffer.ly:.3f}, "
                f"rx={self.joy_stick_buffer.rx:.3f}, "
                f"command={tuple(float(x) for x in self.xyyaw_command.detach().cpu())}"
            )
            self._last_keys = keys

        if self.use_sport_mode:
            if self.joy_stick_buffer.keys & self.WirelessButtons.R1:
                self.get_logger().info("R1 pressed: request sport STANDUP", once=True)
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDUP)
            if self.joy_stick_buffer.keys & self.WirelessButtons.R2:
                self.get_logger().info("R2 pressed: request sport STANDDOWN", once=True)
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDDOWN)
            if self.joy_stick_buffer.keys & self.WirelessButtons.X:
                self.get_logger().info("X pressed: request sport BALANCESTAND", once=True)
                self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)
            if self.joy_stick_buffer.keys & self.WirelessButtons.L1:
                self.get_logger().info(
                    "L1 pressed: disable sport service and enter policy. "
                    f"command={tuple(float(x) for x in self.xyyaw_command.detach().cpu())}"
                )
                self.use_sport_mode = False
                self._sport_state_change(0)
                self.use_parkour_policy = True
                self.reset_policy_state()
                self.send_action(torch.zeros(self.num_actions, device=self.model_device))
                time.sleep(1.0)

        if self.use_parkour_policy:
            loop_start = time.monotonic()
            proprio = self.get_proprio()
            t_proprio = time.monotonic()
            proprio_history = self._get_history_proprio()
            t_history = time.monotonic()
            update_depth = self.global_counter % self.visual_update_interval == 0

            if self.depth_image_buffer is not None:
                action = self.policy_model(
                    proprio,
                    proprio_history,
                    self.depth_image_buffer,
                    update_depth=update_depth,
                )
                t_policy = time.monotonic()
                self.send_action(action)
                t_send = time.monotonic()
                if not self._first_action_sent:
                    self.get_logger().info(
                        "First policy action published: "
                        f"dryrun={self.dryrun}, topic={self.low_cmd_topic}, "
                        f"shape={tuple(action.shape)}, "
                        f"range=({float(action.min().item()):.4f}, {float(action.max().item()):.4f})"
                    )
                    self._first_action_sent = True

                self._debug_log(
                    "step={step} update_depth={update_depth} "
                    "proprio_ms={proprio_ms:.3f} history_ms={history_ms:.3f} "
                    "policy_ms={policy_ms:.3f} send_ms={send_ms:.3f} "
                    "action_min={action_min:.4f} action_max={action_max:.4f} "
                    "depth_shape={depth_shape} depth_finite={depth_finite} "
                    "loop_ms={loop_ms:.3f}".format(
                        step=self.global_counter,
                        update_depth=int(update_depth),
                        proprio_ms=(t_proprio - loop_start) * 1000.0,
                        history_ms=(t_history - t_proprio) * 1000.0,
                        policy_ms=(t_policy - t_history) * 1000.0,
                        send_ms=(t_send - t_policy) * 1000.0,
                        action_min=float(action.min().item()),
                        action_max=float(action.max().item()),
                        depth_shape=tuple(self.depth_image_buffer.shape),
                        depth_finite=bool(torch.isfinite(self.depth_image_buffer).all()),
                        loop_ms=(t_send - loop_start) * 1000.0,
                    )
                )
            else:
                self._debug_log(
                    f"step={self.global_counter} update_depth={int(update_depth)} depth_buffer=missing"
                )

            if self.joy_stick_buffer.keys & self.WirelessButtons.L2:
                self.get_logger().info("L2 pressed: exit policy and request sport mode")
                self.use_parkour_policy = False
                self.use_sport_mode = True
                self._sport_state_change(0)

            if self.joy_stick_buffer.keys & self.WirelessButtons.Y:
                self.get_logger().info("Y pressed: reset policy state")
                self.reset_policy_state()

            self.global_counter += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--nodryrun", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--debug-output",
        type=str,
        default="console",
        choices=["console", "file"],
        help="Where to write debug timing logs.",
    )
    parser.add_argument(
        "--debug-log-path",
        type=str,
        default=None,
        help="Debug log file path when --debug-output=file.",
    )
    args = parser.parse_args()

    with open(osp.join(args.logdir, "config.json"), "r") as f:
        cfg = json.load(f, object_pairs_hook=OrderedDict)

    rclpy.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    debug_logger = None
    if args.debug:
        debug_logger = logging.getLogger("go2_depth_policy_debug")
        debug_logger.setLevel(logging.INFO)
        debug_logger.handlers.clear()
        debug_logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(message)s")
        if args.debug_output == "file":
            debug_log_path = (
                Path(args.debug_log_path)
                if args.debug_log_path is not None
                else Path(args.logdir) / "run_depth_policy.debug.log"
            )
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(debug_log_path, mode="a")
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        debug_logger.addHandler(handler)

    node = Go2DepthPolicyNode(
        "go2",
        cfg=cfg,
        model_device=device,
        dryrun=not args.nodryrun,
        debug=args.debug,
        debug_logger=debug_logger,
    )
    policy, checkpoint_path = load_hardware_vision_policy(args.logdir, cfg, device)
    node.attach_policy(policy)
    node.attach_debug_logger(debug_logger)
    node.get_logger().info(f"Loaded policy from {checkpoint_path}")
    node.get_logger().info(
        "Depth policy node ready: "
        f"device={device}, dryrun={not args.nodryrun}, "
        f"depth_topic=/forward_depth_image, lowcmd_topic={node.low_cmd_topic}"
    )
    node.start_ros_handlers()
    node.start_main_loop_timer(cfg["sim"]["dt"] * cfg["control"]["decimation"])
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
