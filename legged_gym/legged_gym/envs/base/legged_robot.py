# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# ============================================================
# CHANGE SUMMARY (vs previous version)
# ============================================================
# Command logic simplified:
#   - No command curriculum (cfg.commands.curriculum should be False)
#   - has_goal + non-flat envs: vx sampled from [goal_vel_range_min, goal_vel_range_max]
#     (fixed large range, reference extreme parkour style)
#   - flat envs (env_class==17): vx sampled uniformly from full range including 0
#     (to practice fine-grained vel tracking)
#   - no_goal envs: vx sampled from full range including negative (same as before)
#   - _update_command_curriculum removed (no longer called)
#   - _resample_commands simplified: 3-branch logic, no per-env dead zones
#
# Reward functions:
#   - _reward_tracking_goal_vel: clip-type, min(proj_vel, vx_ref)/vx_ref
#     zero when vx_ref<=lin_clip, naturally penalizes backward motion
#   - _reward_stand_still: penalizes actual velocity when cmd~0
#   - _compute_projected_vel_reward: kept as exp-type for episode logging only
#
# Everything else (goals, delta_yaw, terrain curriculum, gap debug) unchanged.
# ============================================================

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gamepad_reader import Gamepad
from scipy.spatial.transform import Rotation as R
from .legged_robot_config import LeggedRobotCfg
from legged_gym.envs.base.depth_noise import DepthNoiseManager

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt


def euler_from_quaternion(quat_angle):
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.curriculum_metric_smooth = 0.0
        self.curriculum_update_alpha = 0.1

        self.resize_transform = torchvision.transforms.Resize(
            (self.cfg.depth.resized[1], self.cfg.depth.resized[0]),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

        if self.cfg.env.joystick_ctrl:
            self.gamepad = Gamepad()
            self.command_function = self.gamepad.get_command
            print("Gamepad control enabled")

        self.noise_manager = DepthNoiseManager(self.cfg, self.device)
        self.csk = None
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.post_physics_step()

    # ------------------------------------------------------------------
    def step(self, actions):
        actions = self.reindex(actions)
        actions.to(self.device)
        self.action_history_buf = torch.cat(
            [self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(
                        self.cfg.domain_rand.action_curr_step.pop(0),
                        device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(
                    self.cfg.domain_rand.action_delay_view,
                    device=self.device, dtype=torch.float)
            indices = -self.delay - 1
            actions = self.action_history_buf[:, indices.long()]

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.extras["delta_yaw_ok"] = torch.abs(self.delta_yaw) < 0.6
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]
        else:
            self.extras["depth"] = None
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_history_observations(self):
        return self.obs_history_buf

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (
            self.cfg.depth.far_clip - self.cfg.depth.near_clip) - 0.5
        return depth_image

    def process_depth_image(self, depth_image, env_id):
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1) - 0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def process_noise_depth_image(self, depth_image, env_id):
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1) - 0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_buf = self.depth_buffer[env_id]
        csk = self.global_counter if self.csk is None else (self.csk + 1)
        depth_image = self.noise_manager.add_noise(depth_image, depth_buf, csk)
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return
        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        for i in range(self.num_envs):
            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.noise_manager.sample_mapping_condition()
            depth_image_ = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[i], self.cam_handles[i], gymapi.IMAGE_DEPTH)
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image_clean = self.process_depth_image(depth_image, i)
            depth_image = self.process_noise_depth_image(depth_image, i)
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
                self.depth_buffer_clean[i] = torch.stack([depth_image_clean] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat(
                    [self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)
                self.depth_buffer_clean[i] = torch.cat(
                    [self.depth_buffer_clean[i, 1:], depth_image_clean.unsqueeze(0)], dim=0)
        self.gym.end_access_image_tensors(self.sim)

    # ------------------------------------------------------------------
    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0

        self.reached_goal_ids = (
            torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1)
            < self.cfg.env.next_goal_threshold)
        self.reach_goal_timer[self.reached_goal_ids] += 1

        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        has_goals = self.env_has_goals
        real_delta_yaw = wrap_to_pi(self.target_yaw - self.yaw)
        fake_delta_yaw = wrap_to_pi(self.fake_target_yaw - self.yaw)
        self.delta_yaw = torch.where(has_goals, real_delta_yaw, fake_delta_yaw)
        self.commands[:, 2] = self.delta_yaw

    # ------------------------------------------------------------------
    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.last_contact_forces[:] = self.contact_forces
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        self._update_goals()
        self._post_physics_step_callback()

        if self.init_done:
            self.extras["debug_env0"] = {
                "dist_to_goal": torch.norm(self.target_pos_rel[0]).item(),
                "cmd_vx":       self.commands[0, 0].item(),
                "delta_yaw":    self.commands[0, 2].item(),
                "actual_vx":    self.base_lin_vel[0, 0].item(),
                "actual_wz":    self.base_ang_vel[0, 2].item(),
                "cur_goal_idx": self.cur_goal_idx[0].item(),
            }

        self.check_termination()
        self.compute_reward()
        self._update_gap_debug_stats()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.update_depth_buffer()
        self.compute_observations()

        self.episode_traveled_distance += torch.norm(self.root_states[:, 7:9], dim=-1) * self.dt
        cmd_speed = torch.abs(self.commands[:, 0])
        self.episode_max_possible_distance += cmd_speed * self.dt
        lin_vel_error = torch.norm(self.base_lin_vel[:, :2] - self.commands[:, :2], dim=1)
        heading_error = torch.abs(self.delta_yaw)
        self.episode_cmd_tracking_error += (lin_vel_error + 0.5 * heading_error) * self.dt
        proj_rew = self._compute_projected_vel_reward()
        self.episode_cmd_tracking_reward += proj_rew
        self.episode_step_count += 1

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self._draw_goals()
            self._draw_feet()
            self._draw_env_bounds(self.lookat_id)
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)
                window_name = "Depth Clean"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                if hasattr(self, "depth_buffer_clean") and self.depth_buffer_clean is not None:
                    img = self.depth_buffer_clean[self.lookat_id, -1].detach().cpu().numpy()
                else:
                    img = self.depth_buffer[self.lookat_id, -1].cpu().numpy()
                cv2.imshow(window_name, img + 0.5)
                cv2.waitKey(1)

    # ------------------------------------------------------------------
    def _update_gap_debug_stats(self):
        gap_mask = (self.env_class == 19)
        if not gap_mask.any():
            return
        x_disp = self.root_states[:, 0] - self.env_origins[:, 0]
        self.episode_max_x_reached = torch.where(
            gap_mask,
            torch.maximum(self.episode_max_x_reached, x_disp),
            self.episode_max_x_reached
        )
        near_gap = gap_mask & (x_disp > 1.5)
        self.episode_near_gap_count += near_gap.float()
        actual_vx = self.base_lin_vel[:, 0]
        retreating = near_gap & (actual_vx < -0.1)
        self.episode_retreat_count += retreating.float()

    def reindex_feet(self, vec):
        return vec[:, [1, 0, 3, 2]]

    def reindex(self, vec):
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def check_termination(self):
        self.reset_buf = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        roll_cutoff    = torch.abs(self.roll) > 1.5
        pitch_cutoff   = torch.abs(self.pitch) > 1.5
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        height_cutoff  = self.root_states[:, 2] < -0.25
        out_of_bounds  = self._check_out_of_bounds()

        self.time_out_buf  = self.episode_length_buf > self.max_episode_length
        self.time_out_buf |= reach_goal_cutoff
        self.time_out_buf |= out_of_bounds

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff

    # ------------------------------------------------------------------
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # NOTE: command curriculum removed; cfg.commands.curriculum should be False

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        _, _, new_yaw = euler_from_quaternion(self.root_states[env_ids, 3:7])
        delta_yaw_range = getattr(self.cfg.commands, 'heading_error_range', [-1.57, 1.57])
        sampled_delta = torch_rand_float(
            delta_yaw_range[0], delta_yaw_range[1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.fake_target_yaw[env_ids] = wrap_to_pi(new_yaw + sampled_delta)

        self.last_actions[env_ids]      = 0.
        self.last_dof_vel[env_ids]      = 0.
        self.last_torques[env_ids]      = 0.
        self.last_root_vel[:]           = 0.
        self.feet_air_time[env_ids]     = 0.
        self.reset_buf[env_ids]         = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.contact_buf[env_ids, :, :]     = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self._cur_goal_idx_before_reset = self.cur_goal_idx.clone()
        self.cur_goal_idx[env_ids]      = 0
        self.reach_goal_timer[env_ids]  = 0
        self.episode_traveled_distance[env_ids]      = 0.
        self.episode_max_possible_distance[env_ids]  = 0.

        self.extras["episode"] = {}
        self.extras["episode_rew"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode_rew"]['rew_' + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s)
            self.episode_sums[key][env_ids] = 0.

        if self.init_done:
            has_goals_mask = self.env_has_goals[env_ids]
            no_goals_mask  = ~has_goals_mask
            cmds     = self.commands[env_ids]
            lin_clip = self.cfg.commands.lin_vel_clip

            self.extras["episode"]["terrain_mean_level"] = torch.mean(self.terrain_levels[env_ids].float())
            # self.extras["episode"]["terrain_has_goals_ratio"] = torch.mean(has_goals_mask.float())
            tc = self.env_class[env_ids]
            # for idx, name in {17:"flat", 18:"step", 19:"gap", 16:"hurdle", 15:"parkour"}.items():
            #     self.extras["episode"][f"terrain_frac_{name}"] = torch.mean((tc == idx).float())

            # if has_goals_mask.any():
            #     gc = cmds[has_goals_mask]
            #     self.extras["episode"]["goals_cmd_mean_vx"] = torch.mean(gc[:, 0])
            #     self.extras["episode"]["goals_cmd_frac_vx_pos"] = torch.mean((gc[:, 0] > lin_clip).float())
            #     goals_done = (self._cur_goal_idx_before_reset[env_ids[has_goals_mask]].float()
            #                   / max(self.cfg.terrain.num_goals, 1))
            #     self.extras["episode"]["goals_completion_rate"] = torch.mean(goals_done)

            # if no_goals_mask.any():
            #     nc = cmds[no_goals_mask]
            #     self.extras["episode"]["nogoals_cmd_mean_vx"] = torch.mean(nc[:, 0])
            #     self.extras["episode"]["nogoals_cmd_frac_vx_pos"] = torch.mean((nc[:, 0] > lin_clip).float())
            #     self.extras["episode"]["nogoals_mean_abs_delta_yaw"] = torch.mean(
            #         torch.abs(self.delta_yaw[env_ids[no_goals_mask]]))

            step_count  = self.episode_step_count[env_ids].clamp(min=1)
            avg_tracking = self.episode_cmd_tracking_reward[env_ids] / step_count
            # self.extras["episode"]["survival_ratio"] = torch.mean(
            #     self.episode_length_buf[env_ids].float() / self.max_episode_length)
            # self.extras["episode"]["avg_tracking_reward"] = torch.mean(avg_tracking)

        self.episode_length_buf[env_ids]          = 0
        self.episode_cmd_tracking_error[env_ids]  = 0
        self.episode_cmd_tracking_reward[env_ids] = 0
        self.episode_step_count[env_ids]          = 0

        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # # gap debug
        # if self.init_done and hasattr(self, 'roll'):
        #     gap_mask_reset = (self.env_class[env_ids] == 19)
        #     if gap_mask_reset.any():
        #         gap_ids = env_ids[gap_mask_reset]
        #         n = len(gap_ids)

        #         fell_in   = self.root_states[gap_ids, 2] < -0.25
        #         rolled    = (torch.abs(self.roll[gap_ids]) > 1.5) | (torch.abs(self.pitch[gap_ids]) > 1.5)
        #         timed_out = self.time_out_buf[gap_ids]

        #         max_x        = self.episode_max_x_reached[gap_ids]
        #         near_cnt     = self.episode_near_gap_count[gap_ids].clamp(min=1)
        #         ret_cnt      = self.episode_retreat_count[gap_ids]
        #         retreat_rate = ret_cnt / near_cnt

        #         self.extras["episode"]["gap/frac_fell_in"]           = fell_in.float().mean()
        #         self.extras["episode"]["gap/frac_rolled"]            = rolled.float().mean()
        #         self.extras["episode"]["gap/frac_timeout"]           = timed_out.float().mean()
        #         self.extras["episode"]["gap/mean_max_x_disp"]        = max_x.mean()
        #         self.extras["episode"]["gap/max_x_p90"]              = max_x.kthvalue(max(1, int(0.9 * n)))[0]
        #         self.extras["episode"]["gap/retreat_rate_near_gap"]  = retreat_rate.mean()
        #         self.extras["episode"]["gap/frac_never_reached_gap"] = (max_x < 1.5).float().mean()

        #         self.episode_max_x_reached[gap_ids]  = 0.
        #         self.episode_near_gap_count[gap_ids] = 0.
        #         self.episode_retreat_count[gap_ids]  = 0.

    # ------------------------------------------------------------------
    def compute_reward(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew  = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_phase_encoding(self):
        t  = self.episode_length_buf.float() * self.dt
        t  = t.unsqueeze(-1)
        f1 = getattr(self.cfg.env, 'f1', 1.0)
        f2 = getattr(self.cfg.env, 'f2', 2.0)
        return torch.cat([
            torch.sin(2 * math.pi * f1 * t),
            torch.cos(2 * math.pi * f1 * t),
            torch.sin(2 * math.pi * f2 * t),
            torch.cos(2 * math.pi * f2 * t),
        ], dim=-1)

    # ------------------------------------------------------------------
    def compute_observations(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)

        if self.cfg.env.joystick_ctrl:
            lin_speed, ang_vel_z, gait_type, e_stop, _ = self.command_function()
            if e_stop:
                import sys; sys.exit(0)
            self.commands[:, 0] = lin_speed[0]
            self.commands[:, 1] = lin_speed[1]

        phase_obs = self.compute_phase_encoding()

        obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            imu_obs,
            self.commands[:, 0:1],
            self.commands[:, 1:2],
            self.commands[:, 2:3],
            0 * self.commands[:, 0:3],
            (self.env_class != 17).float()[:, None],
            (self.env_class == 17).float()[:, None],
            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
            self.reindex(self.action_history_buf[:, -1]),
            self.reindex_feet(self.contact_filt.float() - 0.5),
        ), dim=-1)

        priv_explicit = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            0 * self.base_lin_vel,
            0 * self.base_lin_vel), dim=-1)
        priv_latent = torch.cat((
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1,
            self.motor_strength[1] - 1), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            self.obs_buf = torch.cat(
                [phase_obs, obs_buf, heights, priv_explicit, priv_latent,
                 self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat(
                [phase_obs, obs_buf, priv_explicit, priv_latent,
                 self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1))

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([self.contact_buf[:, 1:], self.contact_filt.float().unsqueeze(1)], dim=1))

    def get_noisy_measurement(self, x, scale):
        if self.cfg.noise.add_noise:
            x = x + (2.0 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level
        return x

    # ------------------------------------------------------------------
    def create_sim(self):
        self.up_axis_idx = 2
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id,
            self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        start = time()
        print("*" * 80)
        print("Start creating ground...")
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if   mesh_type == 'plane':      self._create_ground_plane()
        elif mesh_type == 'heightfield': self._create_heightfield()
        elif mesh_type == 'trimesh':    self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised.")
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*" * 80)
        self._create_envs()

    def set_camera(self, position, lookat):
        cam_pos    = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0],   lookat[1],   lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def set_curriculum_metric(self, value):
        self.curriculum_metric_smooth = (
            self.curriculum_update_alpha * value
            + (1 - self.curriculum_update_alpha) * self.curriculum_metric_smooth)
        self.curriculum_metric = value

    # ------------------------------------------------------------------
    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                friction_range  = self.cfg.domain_rand.friction_range
                num_buckets     = 64
                bucket_ids      = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self.dof_pos_limits  = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device)
            self.dof_vel_limits  = torch.zeros(self.num_dof,    dtype=torch.float, device=self.device)
            self.torque_limits   = torch.zeros(self.num_dof,    dtype=torch.float, device=self.device)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i]    = props["velocity"][i].item()
                self.torque_limits[i]     = props["effort"][i].item()
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass  = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1,))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1,))
        if self.cfg.domain_rand.randomize_base_com:
            rng_com  = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3,))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        return props, np.concatenate([rand_mass, rand_com])

    # ------------------------------------------------------------------
    def _post_physics_step_callback(self):
        if not self.cfg.env.keyboard_ctrl:
            self._update_commands_based_on_goals()
        if self.cfg.terrain.measure_heights:
            if self.global_counter % self.cfg.depth.update_interval == 0:
                self.measured_heights = self._get_heights()
        if (self.cfg.domain_rand.push_robots
                and self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    # ------------------------------------------------------------------
    def _update_commands_based_on_goals(self):
        """Periodic vx/vy resample. delta_yaw maintained by _update_goals every frame."""
        resample_mask = (
            self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0
        )
        resample_ids = resample_mask.nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)

    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(
            1, (self.cur_goal_idx[:, None, None] + future).expand(-1, -1, self.env_goals.shape[-1])
        ).squeeze(1)

    # ------------------------------------------------------------------
    # [SIMPLIFIED] _resample_commands: 3-branch, no per-env dead zones or curriculum
    #
    # Branch 1 - has_goal + non-flat (step/gap/hurdle/parkour):
    #   vx sampled from [goal_vel_min, goal_vel_max] (fixed large range, extreme-parkour style)
    #   15% prob of vx=0 to learn in-place turning before moving
    #
    # Branch 2 - flat (env_class==17), regardless of has_goal:
    #   vx sampled uniformly from [-vx_max, vx_max] including 0
    #   to practice fine-grained speed tracking across all velocities
    #
    # Branch 3 - no_goal (non-flat):
    #   vx sampled from full range including negative
    #   10% prob of vx=0
    #
    # Config params needed:
    #   cfg.commands.goal_vel_min  (e.g. 0.5)  -- min forward speed for parkour terrains
    #   cfg.commands.goal_vel_max  (e.g. 1.5)  -- max forward speed for parkour terrains
    #   cfg.commands.lin_vel_x_max (e.g. 1.5)  -- flat terrain max speed (both directions)
    #   cfg.commands.lin_vel_clip  (e.g. 0.1)  -- dead-zone threshold
    # ------------------------------------------------------------------
    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        lin_clip    = getattr(self.cfg.commands, 'lin_vel_clip', 0.1)
        goal_vmin   = getattr(self.cfg.commands, 'goal_vel_min', 0.5)
        goal_vmax   = getattr(self.cfg.commands, 'goal_vel_max', 1.5)
        flat_vmax   = getattr(self.cfg.commands, 'lin_vel_x_max',
                              self.command_max_ranges["lin_vel_x"][1])

        has_goals   = self.env_has_goals[env_ids]          # (N,) bool
        is_flat     = (self.env_class[env_ids] == 17)      # (N,) bool
        # parkour branch: has goal AND not flat
        is_parkour  = has_goals & (~is_flat)
        # no-goal non-flat branch
        is_nogoal_nf = (~has_goals) & (~is_flat)

        N = len(env_ids)

        # --- sample vx for all envs at once, then override per branch ---
        vx = torch.zeros(N, device=self.device)

        # Branch 1: parkour (has_goal + non-flat) -- fixed large speed range
        if is_parkour.any():
            pk_idx = is_parkour.nonzero(as_tuple=False).flatten()
            zero_mask = torch.rand(len(pk_idx), device=self.device) < 0.15
            pk_vx = torch.empty(len(pk_idx), device=self.device).uniform_(goal_vmin, goal_vmax)
            pk_vx[zero_mask] = 0.0
            vx[pk_idx] = pk_vx

        # Branch 2: flat -- full range including negative, uniform
        if is_flat.any():
            fl_idx = is_flat.nonzero(as_tuple=False).flatten()
            fl_vx  = torch.empty(len(fl_idx), device=self.device).uniform_(0, flat_vmax)
            # dead-zone clip
            fl_vx[torch.abs(fl_vx) < lin_clip] = 0.0
            vx[fl_idx] = fl_vx

        # Branch 3: no-goal non-flat -- full range including negative
        if is_nogoal_nf.any():
            ng_idx  = is_nogoal_nf.nonzero(as_tuple=False).flatten()
            zero_m  = torch.rand(len(ng_idx), device=self.device) < 0.10
            ng_vx   = torch.empty(len(ng_idx), device=self.device).uniform_(-goal_vmax, goal_vmax)
            ng_vx[torch.abs(ng_vx) < lin_clip] = 0.0
            ng_vx[zero_m] = 0.0
            vx[ng_idx] = ng_vx

        self.commands[env_ids, 0] = vx
        # vy always 0 (can extend if needed)
        self.commands[env_ids, 1] = 0.0
        # commands[:,2] = delta_yaw, maintained by _update_goals, not touched here

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        control_type   = self.cfg.control.control_type
        if control_type == "P":
            if not self.cfg.domain_rand.randomize_motor:
                torques = (self.p_gains * (actions_scaled + self.default_dof_pos_all - self.dof_pos)
                           - self.d_gains * self.dof_vel)
            else:
                torques = (self.motor_strength[0] * self.p_gains
                           * (actions_scaled + self.default_dof_pos_all - self.dof_pos)
                           - self.motor_strength[1] * self.d_gains * self.dof_vel)
        elif control_type == "V":
            torques = (self.p_gains * (actions_scaled - self.dof_vel)
                       - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt)
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = (self.default_dof_pos
                                 + torch_rand_float(0., 0.9, (len(env_ids), self.num_dof), device=self.device))
        self.dof_vel[env_ids] = 0.
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(
                    -0.3, 0.3, (len(env_ids), 2), device=self.device)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = (self.cfg.env.rand_yaw_range
                            * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1))
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = (self.cfg.env.rand_pitch_range
                                  * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1))
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(0 * rand_yaw, rand_pitch, rand_yaw)
                self.root_states[env_ids, 3:7] = quat[:, :]
            if self.cfg.env.randomize_start_y:
                self.root_states[env_ids, 1] += (self.cfg.env.rand_y_range
                                                  * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1))
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return
        move_up   = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        move_down = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        if hasattr(self, 'env_has_goals'):
            has_goals = self.env_has_goals[env_ids]
            no_goals  = ~has_goals
            if has_goals.any():
                goals_completed = self.cur_goal_idx[env_ids].float() / max(self.cfg.terrain.num_goals, 1)
                move_up[has_goals]   = goals_completed[has_goals] > 0.8
                move_down[has_goals] = goals_completed[has_goals] < 0.3
            if no_goals.any():
                step_count     = self.episode_step_count[env_ids].clamp(min=1.0)
                avg_rew        = self.episode_cmd_tracking_reward[env_ids] / step_count
                survival_ratio = self.episode_length_buf[env_ids].float() / self.max_episode_length
                can_eval       = survival_ratio >= 0.1
                move_up[no_goals]   = ((avg_rew > 0.7) & (survival_ratio > 0.3) & can_eval)[no_goals]
                move_down[no_goals] = (((avg_rew < 0.4) | (survival_ratio < 0.3)) & can_eval | ~can_eval)[no_goals]
        else:
            step_count     = self.episode_step_count[env_ids].clamp(min=1.0)
            avg_rew        = self.episode_cmd_tracking_reward[env_ids] / step_count
            survival_ratio = self.episode_length_buf[env_ids].float() / self.max_episode_length
            move_up   = (avg_rew > 0.7) & (survival_ratio > 0.3)
            move_down = (avg_rew < 0.4) | (survival_ratio < 0.1)

        self.terrain_levels[env_ids] += move_up.long() - move_down.long()
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0))

        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        if hasattr(self, 'terrain') and hasattr(self.terrain, 'has_goals'):
            levels = self.terrain_levels[env_ids].cpu().numpy().astype(int)
            types  = self.terrain_types[env_ids].cpu().numpy().astype(int)
            self.env_has_goals[env_ids] = torch.from_numpy(
                self.terrain.has_goals[levels, types]).to(self.device).to(torch.bool)
        temp     = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat(
            (temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals  = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        self._update_env_bounds(env_ids)

    # ------------------------------------------------------------------
    def _init_buffers(self):
        actor_root_state      = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor      = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces    = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor_tensor   = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.root_states        = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states  = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state          = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos            = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel            = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat          = self.root_states[:, 3:7]
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6)
        self.contact_forces     = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec   = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec   = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques       = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.p_gains       = torch.zeros(self.num_actions, dtype=torch.float, device=self.device)
        self.d_gains       = torch.zeros(self.num_actions, dtype=torch.float, device=self.device)
        self.actions       = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions  = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel  = torch.zeros_like(self.dof_vel)
        self.last_torques  = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = ((str_rng[1] - str_rng[0])
                               * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
                               + str_rng[0])
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(
                self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio,
                device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(
            self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs,
            device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(
            self.num_envs, self.cfg.env.contact_buf_len, 4,
            device=self.device, dtype=torch.float)

        self.commands       = torch.zeros(self.num_envs, self.cfg.commands.num_commands,
                                          dtype=torch.float, device=self.device)
        self.feet_air_time  = torch.zeros(self.num_envs, self.feet_indices.shape[0],
                                          dtype=torch.float, device=self.device)
        self.last_contacts  = torch.zeros(self.num_envs, len(self.feet_indices),
                                          dtype=torch.bool, device=self.device)
        self.last_contact_forces = torch.zeros_like(self.contact_forces)
        self.base_lin_vel   = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel   = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.default_dof_pos     = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            name  = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.; self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos     = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos_all[:] = self.default_dof_pos[0]

        self.height_update_interval = 1
        if hasattr(self.cfg.env, "height_update_dt"):
            self.height_update_interval = int(
                self.cfg.env.height_update_dt / (self.cfg.sim.dt * self.cfg.control.decimation))

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(
                self.num_envs, self.cfg.depth.buffer_len,
                self.cfg.depth.resized[1], self.cfg.depth.resized[0]).to(self.device)
            self.depth_buffer_clean = torch.zeros(
                self.num_envs, self.cfg.depth.buffer_len,
                self.cfg.depth.resized[1], self.cfg.depth.resized[0]).to(self.device)

        self.env_bounds = torch.zeros(self.num_envs, 4, device=self.device)
        self._compute_env_bounds()

        self.env_has_goals = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if hasattr(self, 'terrain') and hasattr(self.terrain, 'has_goals'):
            if hasattr(self, 'terrain_levels') and hasattr(self, 'terrain_types'):
                levels = self.terrain_levels.cpu().numpy().astype(int)
                types  = self.terrain_types.cpu().numpy().astype(int)
                self.env_has_goals[:] = torch.from_numpy(
                    self.terrain.has_goals[levels, types]).to(self.device).to(torch.bool)

        self.fake_target_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.delta_yaw       = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # episode stats
        self.sampled_x_cmd_buffer         = torch.zeros(self.num_envs, device=self.device)
        self.episode_traveled_distance     = torch.zeros(self.num_envs, device=self.device)
        self.episode_max_possible_distance = torch.zeros(self.num_envs, device=self.device)
        self.episode_cmd_tracking_error    = torch.zeros(self.num_envs, device=self.device)
        self.episode_cmd_tracking_reward   = torch.zeros(self.num_envs, device=self.device)
        self.episode_step_count            = torch.zeros(self.num_envs, device=self.device)

        # command ranges: only used for compatibility lookups (goal_vel_max fallback)
        self.command_max_ranges = class_to_dict(self.cfg.commands.max_ranges)
        # no per-env command_ranges/dead_zones needed since curriculum is removed
        self.command_ranges = {
            "lin_vel_x": torch.tensor(
                [self.command_max_ranges["lin_vel_x"][0],
                 self.command_max_ranges["lin_vel_x"][1]],
                dtype=torch.float, device=self.device).unsqueeze(0).expand(self.num_envs, -1),
            "lin_vel_y": torch.tensor([0., 0.], dtype=torch.float, device=self.device
                         ).unsqueeze(0).expand(self.num_envs, -1),
            "ang_vel_z": torch.tensor([0., 0.], dtype=torch.float, device=self.device
                         ).unsqueeze(0).expand(self.num_envs, -1),
        }

        self._resample_commands(torch.arange(self.num_envs, device=self.device))

        # gap debug buffers
        self.episode_max_x_reached    = torch.zeros(self.num_envs, device=self.device)
        self.episode_min_x_at_death   = torch.zeros(self.num_envs, device=self.device)
        self.episode_vx_when_near_gap = torch.zeros(self.num_envs, device=self.device)
        self.episode_near_gap_count   = torch.zeros(self.num_envs, device=self.device)
        self.episode_retreat_count    = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    def _compute_env_bounds(self):
        border_margin = getattr(self.cfg.env, 'border_margin', 0.1)
        env_length    = self.cfg.terrain.terrain_length
        env_width     = self.cfg.terrain.terrain_width
        row = self.terrain_levels.float()
        col = self.terrain_types.float()
        self.env_bounds[:, 0] = row * env_length + border_margin
        self.env_bounds[:, 1] = (row + 1) * env_length - border_margin
        self.env_bounds[:, 2] = col * env_width + border_margin
        self.env_bounds[:, 3] = (col + 1) * env_width - border_margin

    def _update_env_bounds(self, env_ids):
        border_margin = getattr(self.cfg.env, 'border_margin', 0.1)
        env_length    = self.cfg.terrain.terrain_length
        env_width     = self.cfg.terrain.terrain_width
        row = self.terrain_levels[env_ids].float()
        col = self.terrain_types[env_ids].float()
        self.env_bounds[env_ids, 0] = row * env_length + border_margin
        self.env_bounds[env_ids, 1] = (row + 1) * env_length - border_margin
        self.env_bounds[env_ids, 2] = col * env_width + border_margin
        self.env_bounds[env_ids, 3] = (col + 1) * env_width - border_margin

    def _check_out_of_bounds(self):
        px = self.root_states[:, 0]
        py = self.root_states[:, 1]
        return ((px < self.env_bounds[:, 0]) | (px > self.env_bounds[:, 1])
                | (py < self.env_bounds[:, 2]) | (py > self.env_bounds[:, 3]))

    def _prepare_reward_function(self):
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names     = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, '_reward_' + name))
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in self.reward_scales.keys()}

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales    = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        for rew in self.reward_scales:
            self.reward_scales[rew] = self.reward_scales[rew] / 1
        self.command_max_ranges = class_to_dict(self.cfg.commands.max_ranges)
        self.command_ranges     = None  # will be set in _init_buffers
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length   = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    # ==================================================================
    # Reward functions
    # ==================================================================

    def _compute_projected_vel_reward(self):
        """Helper (exp-type, no scale): used for episode stats / curriculum only."""
        target_yaw = self.yaw + self.delta_yaw
        goal_dir_x = torch.cos(target_yaw)
        goal_dir_y = torch.sin(target_yaw)
        vel_world  = self.root_states[:, 7:9]
        proj_vel   = vel_world[:, 0] * goal_dir_x + vel_world[:, 1] * goal_dir_y
        vx_ref     = self.commands[:, 0]
        return torch.exp(-torch.square(proj_vel - vx_ref) / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_goal_vel(self):
        """
        Clip-type tracking reward.
        - vx_ref <= lin_clip  →  reward = 0  (stand_still handles this case)
        - 0 < proj_vel < vx_ref  →  linear in [0, 1)
        - proj_vel >= vx_ref     →  reward = 1  (capped, no extra for overshooting)
        - proj_vel < 0           →  reward < 0  (natural penalty for going backward)
        """
        target_yaw  = self.yaw + self.delta_yaw
        goal_dir    = torch.stack([torch.cos(target_yaw), torch.sin(target_yaw)], dim=-1)
        proj_vel    = torch.sum(self.root_states[:, 7:9] * goal_dir, dim=-1)
        vx_ref      = self.commands[:, 0]
        lin_clip    = getattr(self.cfg.commands, 'lin_vel_clip', 0.1)

        moving_mask = (vx_ref > lin_clip).float()
        moving_rew  = torch.minimum(proj_vel, vx_ref) / (vx_ref + 1e-5)
        moving_rew  = torch.clamp(moving_rew, max=1.0)  # cap upside; keep negative for backward

        return moving_rew * moving_mask

    def _reward_stand_still(self):
        """
        Penalize actual motion when command is near zero.
        Activated only when vx_ref <= lin_clip (complement of tracking_goal_vel mask).
        """
        lin_clip      = getattr(self.cfg.commands, 'lin_vel_clip', 0.1)
        cmd_near_zero = (torch.norm(self.commands[:, :2], dim=1) < lin_clip).float()
        vel_penalty   = torch.norm(self.root_states[:, 7:9], dim=-1)
        joint_penalty = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * 0.1
        return (vel_penalty + joint_penalty) * cmd_near_zero

    def _reward_tracking_ang_vel_z(self):
        """Heading alignment bonus: exp(-delta_yaw^2/sigma). Small weight."""
        return torch.exp(-torch.square(self.delta_yaw) / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_forward(self):
        forward_mask  = self.commands[:, 0] > 0
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma) * forward_mask.float()

    def _reward_tracking_lin_vel_backward(self):
        backward_mask = self.commands[:, 0] < 0
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma) * backward_mask.float()

    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        rew[self.env_class != 17] *= 0.1
        return rew

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_roll_orientation(self):
        return torch.square(self.projected_gravity[:, 0])

    def _reward_pitch_orientation(self):
        return torch.square(self.projected_gravity[:, 1])

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        return torch.sum(
            1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1)

    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(
            self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_dof_error(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_feet_stumble(self):
        rew = torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2]
                        + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale
                       ).round().long()
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge  = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
        self.feet_at_edge = self.contact_filt & feet_at_edge
        return (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_feet_air_time(self):
        contact      = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            torch.clamp(self.feet_air_time - 0.1, min=0.) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_feet_phase(self):
        num_feet = self.feet_indices.shape[0]
        contact_z_curr  = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_z_last  = self.last_contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt    = torch.logical_or(contact_z_curr, contact_z_last)
        contact_support = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.0
        if not hasattr(self, "feet_stance_time"):
            self.feet_stance_time    = torch.zeros(self.num_envs, num_feet, device=self.device)
        if not hasattr(self, "last_stance_duration"):
            self.last_stance_duration = torch.zeros(self.num_envs, num_feet, device=self.device)
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        if not hasattr(self, "_prev_contact_support"):
            self._prev_contact_support = contact_support.clone()
        lift_off = self._prev_contact_support & (~contact_support)
        self.feet_air_time += self.dt
        self.feet_stance_time[contact_support]  += self.dt
        self.last_stance_duration[lift_off]      = self.feet_stance_time[lift_off]
        self.feet_stance_time[~contact_support]  = 0.0
        lin_vel_cmd = torch.norm(self.commands[:, :2], dim=1)
        ratio_at_zero = getattr(self.cfg.rewards, "stance_ratio_at_low_speed", 0.7)
        ratio_at_max  = getattr(self.cfg.rewards, "stance_ratio_at_high_speed", 0.4)
        max_speed     = getattr(self.cfg.commands, 'goal_vel_max', 1.5)
        min_speed     = getattr(self.cfg.commands, "lin_vel_clip", 0.1)
        speed_norm    = (torch.clamp(lin_vel_cmd, min_speed, max_speed) - min_speed) / (max_speed - min_speed)
        desired_ratio = ratio_at_zero - (ratio_at_zero - ratio_at_max) * torch.sigmoid(2.0*(speed_norm - 0.5))
        desired_ratio_per_foot = desired_ratio[:, None].expand(-1, num_feet)
        swing    = self.feet_air_time
        stance   = self.last_stance_duration
        cycle    = stance + swing + 1e-6
        ratio_err = torch.abs(stance / cycle - desired_ratio_per_foot)
        min_cycle_time = getattr(self.cfg.rewards, "min_cycle_time", 0.5)
        cycle_penalty  = torch.clamp(min_cycle_time - cycle, min=0.0)
        reward = torch.sum((ratio_err + cycle_penalty) * first_contact.float(), dim=1)
        lin_vel_clip = getattr(self.cfg.commands, "lin_vel_clip", 0.1)
        reward *= (torch.norm(self.commands[:, :2], dim=1) > lin_vel_clip).float()
        self.feet_air_time  *= (~contact_filt).float()
        self._prev_contact_support = contact_support.clone()
        return reward

    def _reward_feet_contact_balance(self):
        contact_ratio = self.contact_buf.mean(dim=1)
        return torch.var(contact_ratio, dim=1)

    def _reward_lazy_stop(self):
        lin_vel_clip  = getattr(self.cfg.commands, 'lin_vel_clip', 0.1)
        lin_vel_error = torch.norm(self.base_lin_vel[:, :2] - self.commands[:, :2], dim=1)
        lin_cmd_nonzero = torch.norm(self.commands[:, :2], dim=1) > lin_vel_clip
        lin_penalty = torch.square(lin_vel_error) * lin_cmd_nonzero.float()

        kp = getattr(self.cfg.commands, 'lazy_stop_wz_kp', 1.0)
        ang_vel_z_max = self.command_max_ranges["ang_vel_z"][1]
        expected_wz   = torch.clamp(kp * self.delta_yaw, -ang_vel_z_max, ang_vel_z_max)
        ang_penalty   = (torch.square(self.base_ang_vel[:, 2] - expected_wz)
                         * (torch.abs(self.delta_yaw) > 0.3).float())
        return lin_penalty + 0.5 * ang_penalty

    def _reward_base_height(self):
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_dof_error_max(self):
        per_joint = torch.abs(self.dof_pos - self.default_dof_pos)
        # 取每个环境中偏差最大的那个关节
        return torch.max(per_joint, dim=1)[0]

    # ==================================================================
    # Visualization helpers
    # ==================================================================

    def _draw_height_samples(self):
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        i = self.lookat_id
        base_pos = self.root_states[i, :3].cpu().numpy()
        heights  = self.measured_heights[i].cpu().numpy()
        height_points = quat_apply_yaw(
            self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, heights[j]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_goals(self):
        if hasattr(self, "env_has_goals") and not self.env_has_goals[self.lookat_id]:
            return
        sphere_geom_passed  = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0,1,0))
        sphere_geom_cur     = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0,0,1))
        sphere_geom_next    = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1,0,0))
        sphere_geom_default = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0.6,0.6,0.6))
        sphere_geom_reached = gymutil.WireframeSphereGeometry(
            self.cfg.env.next_goal_threshold, 32, 32, None, color=(0,1,0))
        goals   = self.terrain_goals[self.terrain_levels[self.lookat_id],
                                     self.terrain_types[self.lookat_id]].cpu().numpy()
        cur_idx = int(self.cur_goal_idx[self.lookat_id].cpu().item())
        n_goals = goals.shape[0]
        for i, goal in enumerate(goals):
            goal_xy = goal[:2] + self.terrain.cfg.border_size
            pts     = (goal_xy / self.terrain.cfg.horizontal_scale).astype(int)
            goal_z  = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale
            pose    = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
            if i < cur_idx:
                gymutil.draw_lines(sphere_geom_passed,  self.gym, self.viewer, self.envs[self.lookat_id], pose)
            elif i == cur_idx:
                gymutil.draw_lines(sphere_geom_cur,     self.gym, self.viewer, self.envs[self.lookat_id], pose)
                if self.reached_goal_ids[self.lookat_id]:
                    gymutil.draw_lines(sphere_geom_reached, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            elif i == cur_idx + 1 and (cur_idx + 1) < n_goals:
                gymutil.draw_lines(sphere_geom_next,    self.gym, self.viewer, self.envs[self.lookat_id], pose)
            else:
                gymutil.draw_lines(sphere_geom_default, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        if not self.cfg.depth.use_camera:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1,0.35,0.25))
            pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
            for i in range(5):
                norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def _draw_feet(self):
        if hasattr(self, 'feet_at_edge'):
            non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0,1,0))
            edge_geom     = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1,0,0))
            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            for fi in range(self.feet_indices.shape[0]):
                pose = gymapi.Transform(gymapi.Vec3(
                    feet_pos[self.lookat_id, fi, 0],
                    feet_pos[self.lookat_id, fi, 1],
                    feet_pos[self.lookat_id, fi, 2]), r=None)
                geom = edge_geom if self.feet_at_edge[self.lookat_id, fi] else non_edge_geom
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def _draw_env_bounds(self, env_id=None):
        if not hasattr(self, "env_bounds") or self.env_bounds is None:
            return
        if env_id is None:
            env_id = self.lookat_id
        try:
            b = self.env_bounds[env_id].cpu().numpy()
            x_min, x_max, y_min, y_max = b
        except Exception:
            return
        z          = float(self.env_origins[env_id, 2].cpu().item()) + 0.05
        env_handle = self.envs[env_id]
        corners    = np.array([[x_min,y_min,z],[x_max,y_min,z],[x_max,y_max,z],[x_min,y_max,z]], dtype=np.float32)
        line_indices = np.array([[0,1],[1,2],[2,3],[3,0]], dtype=np.int32)
        class SimpleLineGeom:
            def __init__(s, pts, idx, col):
                s.points=np.array(pts,dtype=np.float32); s.indices=np.array(idx,dtype=np.int32)
                s._colors=np.tile(np.array(col,dtype=np.float32),(len(s.indices),1))
            def num_lines(s): return len(s.indices)
            def colors(s): return s._colors
            def instance_verts(s, pose):
                t=np.array([pose.p.x,pose.p.y,pose.p.z],dtype=np.float32)
                v=np.empty((len(s.indices)*2,3),dtype=np.float32)
                for i,(a,b) in enumerate(s.indices): v[2*i]=s.points[a]+t; v[2*i+1]=s.points[b]+t
                return v
        try:
            gymutil.draw_lines(SimpleLineGeom(corners, line_indices, [1.,0.,0.]),
                               self.gym, self.viewer, env_handle, gymapi.Transform())
        except Exception:
            pass

    def _init_height_points(self):
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y)
        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device)
        for i in range(self.num_envs):
            offset   = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise,
                                         self.cfg.terrain.measure_horizontal_noise,
                                         (self.num_height_points, 2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise,
                                         self.cfg.terrain.measure_horizontal_noise,
                                         (self.num_height_points, 2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points

    def _get_heights(self, env_ids=None):
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")
        if env_ids:
            points = (quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                     self.height_points[env_ids])
                      + self.root_states[env_ids, :3].unsqueeze(1))
        else:
            points = (quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points),
                                     self.height_points)
                      + self.root_states[:, :3].unsqueeze(1))
        points += self.terrain.cfg.border_size
        points  = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[:,:,0].view(-1), 0, self.height_samples.shape[0]-2)
        py = torch.clip(points[:,:,1].view(-1), 0, self.height_samples.shape[1]-2)
        heights = torch.min(torch.min(self.height_samples[px,py],
                                      self.height_samples[px+1,py]),
                            self.height_samples[px,py+1])
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    # ------------------------------------------------------------------
    # Gym env creation helpers (unchanged)
    # ------------------------------------------------------------------

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal           = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction  = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution      = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale    = self.cfg.terrain.horizontal_scale
        hf_params.row_scale       = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale  = self.cfg.terrain.vertical_scale
        hf_params.nbRows          = self.terrain.tot_cols
        hf_params.nbColumns       = self.terrain.tot_rows
        hf_params.transform.p.x   = -self.terrain.border
        hf_params.transform.p.y   = -self.terrain.border
        hf_params.transform.p.z   = 0.0
        hf_params.static_friction  = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution      = self.cfg.terrain.restitution
        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.flatten(order='C'), hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(
            self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices      = self.terrain.vertices.shape[0]
        tm_params.nb_triangles     = self.terrain.triangles.shape[0]
        tm_params.transform.p.x    = -self.terrain.cfg.border_size
        tm_params.transform.p.y    = -self.terrain.cfg.border_size
        tm_params.transform.p.z    = 0.0
        tm_params.static_friction  = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution      = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order='C'),
            self.terrain.triangles.flatten(order='C'), tm_params)
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(
            self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(
            self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config        = self.cfg.depth
            camera_props  = gymapi.CameraProperties()
            camera_props.width          = config.original[0]
            camera_props.height         = config.original[1]
            camera_props.enable_tensors = True
            if hasattr(config, "near_plane"):
                camera_props.near_plane = config.near_plane
            camera_horizontal_fov = config.horizontal_fov
            if isinstance(camera_horizontal_fov, (tuple, list)):
                camera_props.horizontal_fov = np.random.uniform(
                    camera_horizontal_fov[0], camera_horizontal_fov[1])
            else:
                camera_props.horizontal_fov = camera_horizontal_fov
            camera_handle   = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)
            local_transform = gymapi.Transform()
            if isinstance(config.position, dict):
                cam_x = np.random.normal(config.position['mean'][0], config.position['std'][0])
                cam_y = np.random.normal(config.position['mean'][1], config.position['std'][1])
                cam_z = np.random.normal(config.position['mean'][2], config.position['std'][2])
                local_transform.p = gymapi.Vec3(cam_x, cam_y, cam_z)
            else:
                local_transform.p = gymapi.Vec3(*np.copy(config.position))
            if hasattr(config, "rotation"):
                if isinstance(config.rotation, dict):
                    cam_roll  = np.random.uniform(0,1)*(config.rotation["upper"][0]-config.rotation["lower"][0])+config.rotation["lower"][0]
                    cam_pitch = np.random.uniform(0,1)*(config.rotation["upper"][1]-config.rotation["lower"][1])+config.rotation["lower"][1]
                    cam_yaw   = np.random.uniform(0,1)*(config.rotation["upper"][2]-config.rotation["lower"][2])+config.rotation["lower"][2]
                    local_transform.r = gymapi.Quat.from_euler_zyx(cam_roll, cam_pitch, cam_yaw)
            else:
                camera_angle  = np.random.uniform(config.angle[0], config.angle[1])
                local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
            self.gym.attach_camera_to_body(
                camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode         = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints           = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule   = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments         = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link                   = self.cfg.asset.fix_base_link
        asset_options.density                         = self.cfg.asset.density
        asset_options.angular_damping                 = self.cfg.asset.angular_damping
        asset_options.linear_damping                  = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity            = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity             = self.cfg.asset.max_linear_velocity
        asset_options.armature                        = self.cfg.asset.armature
        asset_options.thickness                       = self.cfg.asset.thickness
        asset_options.disable_gravity                 = self.cfg.asset.disable_gravity

        robot_asset        = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof       = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies    = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset    = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        body_names      = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names  = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs   = len(self.dof_names)
        feet_names      = [s for s in body_names if self.cfg.asset.foot_name in s]

        for s in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
            feet_idx   = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (self.cfg.init_state.pos + self.cfg.init_state.rot
                                + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel)
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose   = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles  = []
        self.envs           = []
        self.cam_handles    = []
        self.cam_tensors    = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)

        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(
                    0., 0., self.cfg.env.rand_yaw_range * np.random.uniform(-1, 1))
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)
            self.attach_camera(i, env_handle, anymal_handle)
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i])

        for joint_list, attr in [
            (["FR_hip_joint","FL_hip_joint","RR_hip_joint","RL_hip_joint"], "hip_indices"),
            (["FR_thigh_joint","FL_thigh_joint","RR_thigh_joint","RL_thigh_joint"], "thigh_indices"),
            (["FR_calf_joint","FL_calf_joint","RR_calf_joint","RL_calf_joint"], "calf_indices"),
        ]:
            idx_tensor = torch.zeros(len(joint_list), dtype=torch.long, device=self.device)
            for i, name in enumerate(joint_list):
                idx_tensor[i] = self.dof_names.index(name)
            setattr(self, attr, idx_tensor)

    def _get_env_origins(self):
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
            self.env_class   = torch.zeros(self.num_envs,    device=self.device)
            max_init_level   = min(
                self.cfg.terrain.max_init_terrain_level, self.cfg.terrain.num_rows - 1)
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels  = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types   = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level  = self.cfg.terrain.num_rows
            self.terrain_origins    = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:]     = self.terrain_origins[self.terrain_levels, self.terrain_types]
            self.terrain_class      = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:]       = self.terrain_class[self.terrain_levels, self.terrain_types]
            self.terrain_goals      = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            self.env_goals          = torch.zeros(
                self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3,
                device=self.device)
            self.cur_goal_idx       = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            temp     = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat(
                (temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
            self.cur_goals  = self._gather_cur_goals()
            self.next_goals = self._gather_cur_goals(future=1)
        else:
            self.custom_origins = False
            self.env_origins    = torch.zeros(self.num_envs, 3, device=self.device)
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy   = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing  = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.