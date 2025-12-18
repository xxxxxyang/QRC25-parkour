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
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
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
     
        return roll_x, pitch_y, yaw_z # in radians

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.curriculum_metric_smooth = 0.0
        self.curriculum_update_alpha = 0.1

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
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
        self.csk = None  # current step count for noise manager
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.post_physics_step()

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = self.reindex(actions)

        actions.to(self.device)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
            indices = -self.delay -1
            actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms

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
        self.extras["delta_yaw_ok"] = self.delta_yaw < 0.6
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_history_observations(self):
        return self.obs_history_buf

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image

    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        # print("Processed Depth Image: ", depth_image)
        return depth_image

    def process_noise_depth_image(self, depth_image, env_id):
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        # 把当前 env 的 depth_buffer 传进去
        depth_buf = self.depth_buffer[env_id]
        csk = self.global_counter if self.csk is None else (self.csk + 1)
        depth_image = self.noise_manager.add_noise(depth_image, depth_buf, csk)
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.noise_manager.sample_mapping_condition()
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            depth_image = gymtorch.wrap_tensor(depth_image_)
            # print("Raw Depth Image: ", depth_image)
            depth_image_clean = self.process_depth_image(depth_image, i)
            depth_image = self.process_noise_depth_image(depth_image, i)

            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
                self.depth_buffer_clean[i] = torch.stack([depth_image_clean] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)
                self.depth_buffer_clean[i] = torch.cat([self.depth_buffer_clean[i, 1:], depth_image_clean.unsqueeze(0)], dim=0)

        self.gym.end_access_image_tensors(self.sim)

    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0

        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1

        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
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
        
        # self._update_jump_schedule()
        self._update_goals()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.update_depth_buffer()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.episode_traveled_distance += torch.norm(self.root_states[:, 7:9], dim=-1) * self.dt
        # 累积理论最大可达距离（基于命令速度幅值）
        cmd_speed = torch.norm(self.commands[:, :2], dim=-1)
        self.episode_max_possible_distance += cmd_speed * self.dt
        # 新增：累积命令跟踪误差（线速度 + 角速度）
        lin_vel_error = torch.norm(self.base_lin_vel[:, :2] - self.commands[:, :2], dim=1)
        ang_vel_error = torch.abs(self.base_ang_vel[:, 2] - self.commands[:, 2])
        self.episode_cmd_tracking_error += (lin_vel_error + 0.5 * ang_vel_error) * self.dt
        # 累积跟踪奖励（使用 exp 形式，与 reward 一致）
        tracking_sigma = self.cfg.rewards.tracking_sigma
        lin_tracking_rew = torch.exp(-torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1) / tracking_sigma)
        ang_tracking_rew = torch.exp(-torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2]) / tracking_sigma)
        self.episode_cmd_tracking_reward += (lin_tracking_rew + ang_tracking_rew) * 0.5
        self.episode_step_count += 1

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            # self._draw_height_samples()
            self._draw_goals()
            self._draw_feet()
            self._draw_env_bounds(self.lookat_id)
            # self._draw_commands(self.lookat_id)
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                # cv2.imshow("Depth Image", (self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)*0)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                # print("Depth Image: ", self.depth_buffer[self.lookat_id, -1].cpu().numpy()+0.5)
                cv2.waitKey(1)

                window_name = "Depth Clean"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                img = None
                if hasattr(self, "depth_buffer_clean") and self.depth_buffer_clean is not None:
                    img = self.depth_buffer_clean[self.lookat_id, -1].detach().cpu().numpy()
                else:
                    # 回退到 buffer 的最后一帧（在极少数情况 update 尚未写入时）
                    img = self.depth_buffer[self.lookat_id, -1].cpu().numpy()
                cv2.imshow(window_name, img + 0.5)
                cv2.waitKey(1)

    def reindex_feet(self, vec):
        return vec[:, [1, 0, 3, 2]]

    def reindex(self, vec):
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        roll_cutoff = torch.abs(self.roll) > 1.5
        pitch_cutoff = torch.abs(self.pitch) > 1.5
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        height_cutoff = self.root_states[:, 2] < -0.25
        out_of_bounds = self._check_out_of_bounds() # 边界监测，视为超时不惩罚

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf |= reach_goal_cutoff
        self.time_out_buf |= out_of_bounds

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            self._update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0
        self.episode_traveled_distance[env_ids] = 0.
        self.episode_max_possible_distance[env_ids] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(self.command_ranges["lin_vel_x"][:, 1]).item()
            self.extras["episode"]["max_command_y"] = torch.mean(self.command_ranges["lin_vel_y"][:, 1]).item()
            self.extras["episode"]["max_command_ang_vel"] = torch.mean(self.command_ranges["ang_vel_z"][:, 1]).item()
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        if hasattr(self, 'lin_vel_tracking_reward_avg'):
            self.lin_vel_tracking_reward_avg[env_ids] = 0.0
            # 有 goals 的环境重置后直接启用角速度，无 goals 的先禁用
            if hasattr(self, 'env_has_goals'):
                self.ang_vel_enabled[env_ids] = self.env_has_goals[env_ids]
            else:
                self.ang_vel_enabled[env_ids] = False
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_phase_encoding(self):
        """
        计算基于时间的傅里叶相位编码
        使用 episode 内的时间，在真实机器人上可通过计时器获取
        Returns:
            phase_obs: (num_envs, 4) - 包含 sin/cos 编码的相位观测
        """
        # episode 内的时间（秒）
        t = self.episode_length_buf.float() * self.dt  # shape: (num_envs,) (seconds)
        t = t.unsqueeze(-1)  # shape: (num_envs, 1)
        
        # 步态相关的频率（Hz）- 可在 config 中配置
        f1 = getattr(self.cfg.env, 'f1', 1.0)   # 主步态频率，约 1-2 Hz
        f2 = getattr(self.cfg.env, 'f2', 2.0)   # 高频分量
        
        # 计算傅里叶基
        phase_obs = torch.cat([
            torch.sin(2 * math.pi * f1 * t),
            torch.cos(2 * math.pi * f1 * t),
            torch.sin(2 * math.pi * f2 * t),
            torch.cos(2 * math.pi * f2 * t),
        ], dim=-1)  # shape: (num_envs, 4)
        
        return phase_obs
    
    def compute_observations(self):
        """ 
        Computes observations
        """
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)

        if self.cfg.env.joystick_ctrl:
            # TODO： need to be modified for target yaw
            # use joystick(gamepad) control
            lin_speed, ang_vel_z, gait_type, e_stop, _ = self.command_function()
            if e_stop:
                import sys
                sys.exit(0)
            self.commands[:, 0] = lin_speed[0]
            self.commands[:, 1] = lin_speed[1]
            self.commands[:, 2] = ang_vel_z
        else:
            self.delta_yaw = wrap_to_pi(self.target_yaw - self.yaw)
            self.delta_next_yaw = wrap_to_pi(self.next_target_yaw - self.yaw)

        # 计算相位编码
        phase_obs = self.compute_phase_encoding()  # (num_envs, 4)

        obs_buf = torch.cat((#skill_vector, 
                            self.base_ang_vel  * self.obs_scales.ang_vel,   # [1,3]
                            imu_obs,    # [1,2] roll, pitch
                            self.commands[:, 0:1],  # [1,1] vx
                            self.commands[:, 1:2],  # [1,1] vy
                            self.commands[:, 2:3],  # [1,1] wz
                            0*self.commands[:, 0:3],  # [1,3] 占位
                            (self.env_class != 17).float()[:, None],
                            (self.env_class == 17).float()[:, None],
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            self.reindex_feet(self.contact_filt.float()-0.5),
                            ),dim=-1)
        priv_explicit = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                   0 * self.base_lin_vel,
                                   0 * self.base_lin_vel), dim=-1)
        priv_latent = torch.cat((
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1, 
            self.motor_strength[1] - 1
        ), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            self.obs_buf = torch.cat([phase_obs, obs_buf, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([phase_obs, obs_buf, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        # obs_buf[:, 6:8] = 0  # mask yaw in proprioceptive history
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )
        
        
    def get_noisy_measurement(self, x, scale):
        if self.cfg.noise.add_noise:
            x = x + (2.0 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level
        return x

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id  # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        start = time()
        print("*"*80)
        print("Start creating ground...")
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def set_curriculum_metric(self, value):
        self.curriculum_metric_smooth = (
                                        self.curriculum_update_alpha * value + 
                                        (1 - self.curriculum_update_alpha) * self.curriculum_metric_smooth )
        self.curriculum_metric = value

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # No need to use tensors as only called upon env creation
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1, ))
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self._update_commands_based_on_goals()
        
        if self.cfg.terrain.measure_heights:
            if self.global_counter % self.cfg.depth.update_interval == 0:
                self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    # def _update_commands_based_on_goals(self):
    #     """
    #     根据当前地形是否有 goals 来更新命令：
    #     - 有 goals：基于目标位置生成命令
    #     - 无 goals：使用随机采样的命令
    #     """
    #     # 获取有 goals 和无 goals 的环境 mask
    #     has_goals_mask = self.env_has_goals
    #     no_goals_mask = ~has_goals_mask
        
    #     # 对于有 goals 的环境，基于目标生成命令
    #     if has_goals_mask.any():
    #         self._update_goal_based_commands(has_goals_mask)
        
    #     # 对于无 goals 的环境，按时间间隔重采样随机命令
    #     if no_goals_mask.any():
    #         resample_mask = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0) & no_goals_mask
    #         resample_ids = resample_mask.nonzero(as_tuple=False).flatten()
    #         if len(resample_ids) > 0:
    #             self._resample_commands(resample_ids)
    def _update_commands_based_on_goals(self):
        """
        根据当前地形是否有 goals 来更新命令：
        - 有 goals：基于目标位置生成命令，始终启用角速度
        - 无 goals：使用随机采样的命令，根据线速度跟踪质量动态启用/禁用角速度
        角速度的范围仍由 _update_command_curriculum 根据 terrain level 控制
        """
        # ===== 1. 获取有 goals 和无 goals 的环境 mask =====
        has_goals_mask = self.env_has_goals
        no_goals_mask = ~has_goals_mask
        # ===== 2. 对于无 goals 的环境，更新线速度跟踪奖励的滑动平均 =====
        if no_goals_mask.any():
            lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
            lin_vel_reward = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

            alpha = getattr(self.cfg.commands, 'tracking_avg_alpha', 0.05)
            # 只更新无 goals 环境的滑动平均
            self.lin_vel_tracking_reward_avg[no_goals_mask] = (
                alpha * lin_vel_reward[no_goals_mask] + 
                (1 - alpha) * self.lin_vel_tracking_reward_avg[no_goals_mask]
            )
            # ===== 3. 根据跟踪质量更新角速度启用状态（滞回逻辑，仅对无 goals 环境）=====
            enable_threshold = getattr(self.cfg.commands, 'ang_vel_enable_threshold', 0.6)
            disable_threshold = getattr(self.cfg.commands, 'ang_vel_disable_threshold', 0.3)

            should_enable = (self.lin_vel_tracking_reward_avg > enable_threshold) & no_goals_mask
            should_disable = (self.lin_vel_tracking_reward_avg < disable_threshold) & no_goals_mask

            self.ang_vel_enabled = torch.where(should_enable, torch.ones_like(self.ang_vel_enabled), self.ang_vel_enabled)
            self.ang_vel_enabled = torch.where(should_disable, torch.zeros_like(self.ang_vel_enabled), self.ang_vel_enabled)
        # ===== 4. 有 goals 的环境始终启用角速度 =====
        self.ang_vel_enabled[has_goals_mask] = True
        # ===== 5. 对于有 goals 的环境，基于目标生成命令 =====
        if has_goals_mask.any():
            self._update_goal_based_commands(has_goals_mask)
        # ===== 6. 对于无 goals 的环境，按时间间隔重采样随机命令 =====
        if no_goals_mask.any():
            resample_mask = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0) & no_goals_mask
            resample_ids = resample_mask.nonzero(as_tuple=False).flatten()
            if len(resample_ids) > 0:
                self._resample_commands(resample_ids)
        # ===== 7. 根据角速度启用状态，将未启用的环境角速度置零 =====
        # 注意：角速度的范围由 _update_command_curriculum 控制，这里只是决定是否启用
        self.commands[:, 2] = torch.where(
            self.ang_vel_enabled,
            self.commands[:, 2],
            torch.zeros_like(self.commands[:, 2])
        )

    def _update_goal_based_commands(self, env_mask):
        """
        基于当前目标位置生成命令（仅对 env_mask 为 True 的环境）
        commands: [vx, vy, wz]
        """
        # 将目标相对位置转换到机器人局部坐标系
        target_local = quat_rotate_inverse(self.base_quat, 
                                           torch.cat([self.target_pos_rel, 
                                                     torch.zeros(self.num_envs, 1, device=self.device)], dim=1))
        
        # 计算目标方向的 yaw 角（用于生成角速度命令）
        target_yaw_local = torch.atan2(target_local[:, 1], target_local[:, 0])
        
        # 获取配置参数
        x_ratio = getattr(self.cfg.commands, 'goal_x_ratio', 1.0)
        y_ratio = getattr(self.cfg.commands, 'goal_y_ratio', 0.5)
        yaw_ratio = getattr(self.cfg.commands, 'goal_yaw_ratio', 1.0)
        
        # 计算 vx 命令 (per-env clip)
        x_cmd = torch.clip(
            target_local[:, 0] * x_ratio,
            min=self.command_ranges["lin_vel_x"][:, 0],  # shape: (num_envs,)
            max=self.command_ranges["lin_vel_x"][:, 1],  # shape: (num_envs,)
        )
        
        # 计算 vy 命令 (per-env clip)
        y_cmd = torch.clip(
            target_local[:, 1] * y_ratio,
            min=self.command_ranges["lin_vel_y"][:, 0],
            max=self.command_ranges["lin_vel_y"][:, 1],
        )
        
        # 计算 wz 命令（角速度）(per-env clip)
        wz_cmd = torch.clip(
            target_yaw_local * yaw_ratio,
            min=self.command_ranges["ang_vel_z"][:, 0],
            max=self.command_ranges["ang_vel_z"][:, 1],
        )
        
        # 应用命令截断
        lin_vel_clip = getattr(self.cfg.commands, 'lin_vel_clip', 0.2)
        ang_vel_clip = getattr(self.cfg.commands, 'ang_vel_clip', 0.1)
        
        x_cmd = torch.where(torch.abs(x_cmd) < lin_vel_clip, torch.zeros_like(x_cmd), x_cmd)
        y_cmd = torch.where(torch.abs(y_cmd) < lin_vel_clip, torch.zeros_like(y_cmd), y_cmd)
        wz_cmd = torch.where(torch.abs(wz_cmd) < ang_vel_clip, torch.zeros_like(wz_cmd), wz_cmd)
        
        # 可选：当偏差过大时停止前进
        x_stop_by_yaw_threshold = getattr(self.cfg.commands, 'x_stop_by_yaw_threshold', None)
        if x_stop_by_yaw_threshold is not None:
            large_yaw_mask = torch.abs(target_yaw_local) > x_stop_by_yaw_threshold
            x_cmd = torch.where(large_yaw_mask & env_mask, torch.zeros_like(x_cmd), x_cmd)
        
        # 仅更新有 goals 的环境
        self.commands[env_mask, 0] = x_cmd[env_mask]
        self.commands[env_mask, 1] = y_cmd[env_mask]
        self.commands[env_mask, 2] = wz_cmd[env_mask]
        
    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]+future).expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)

    def _resample_commands(self, env_ids):
        """
        分段采样命令，但确保0值有采样概率
        """
        for idx in env_ids:
            i = int(idx)
            for cmd_idx, cmd_name in enumerate(["lin_vel_x", "lin_vel_y", "ang_vel_z"]):
                # 获取边界
                neg_outer = float(self.command_ranges[cmd_name][i, 0])
                pos_outer = float(self.command_ranges[cmd_name][i, 1])
                neg_inner = float(self.command_dead_zones[cmd_name][i, 0])
                pos_inner = float(self.command_dead_zones[cmd_name][i, 1])
                # === 关键修改：为0值分配固定概率 ===
                # 设置0值的采样概率（10%）
                zero_prob = 0.1
                # 随机决定是否采样0值
                if torch.rand(1, device=self.device).item() < zero_prob:
                    self.commands[i, cmd_idx] = 0.0
                    continue
                # === 原有的分段采样逻辑 ===
                neg_len = abs(neg_inner - neg_outer)
                pos_len = abs(pos_outer - pos_inner)
                total_len = neg_len + pos_len
                if total_len < 1e-6:
                    self.commands[i, cmd_idx] = torch.empty(1, device=self.device).uniform_(
                        neg_outer, pos_outer
                    ).squeeze()
                else:
                    rand_val = torch.rand(1, device=self.device).item()
                    if rand_val < (neg_len / total_len):
                        self.commands[i, cmd_idx] = torch.empty(1, device=self.device).uniform_(
                            neg_outer, neg_inner
                        ).squeeze()
                    else:
                        self.commands[i, cmd_idx] = torch.empty(1, device=self.device).uniform_(
                            pos_inner, pos_outer
                        ).squeeze()
        # 原有的clip逻辑
        self.commands[env_ids, 2] *= torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_clip
        self.commands[env_ids, :2] *= torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip

    def _resample_lin_commands(self, env_ids):
        """Resample linear velocity commands (vx, vy)"""
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device
        ).squeeze(1)
        # set small commands to zero
        self.commands[env_ids, :2] *= torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_motor:  # TODO add strength to gain directly
                torques = self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains*self.dof_vel
            else:
                torques = self.motor_strength[0] * self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[1] * self.d_gains*self.dof_vel
                
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.9, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = self.cfg.env.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = self.cfg.env.rand_pitch_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(0*rand_yaw, rand_pitch, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]  
            if self.cfg.env.randomize_start_y:
                self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """
        分场景的 terrain curriculum：
        - 有 goals：基于目标完成率
        - 无 goals：基于命令跟踪质量（而非存活时间或距离）
        """
        if not self.init_done:
            return

        move_up = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        move_down = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        
        if hasattr(self, 'env_has_goals'):
            has_goals = self.env_has_goals[env_ids]
            no_goals = ~has_goals

            # === 有 goals 的环境：基于目标完成率 ===
            if has_goals.any():
                goals_completed = self.cur_goal_idx[env_ids].float() / max(self.cfg.terrain.num_goals, 1)
                move_up[has_goals] = goals_completed[has_goals] > 0.6
                move_down[has_goals] = goals_completed[has_goals] < 0.2
            
            # === 无 goals 的环境：基于命令跟踪质量 ===
            if no_goals.any():
                step_count = self.episode_step_count[env_ids]
                step_count_safe = torch.clamp(step_count, min=1.0)
                
                # 方法1：平均跟踪奖励（0~1 之间，越高越好）
                avg_tracking_reward = self.episode_cmd_tracking_reward[env_ids] / step_count_safe
                
                # 方法2：平均跟踪误差（越低越好）
                avg_tracking_error = self.episode_cmd_tracking_error[env_ids] / step_count_safe
                
                # 方法3：综合考虑存活比例（但权重很小，避免 lazy_stop 问题）
                survival_ratio = self.episode_length_buf[env_ids].float() / self.max_episode_length
                
                # 核心指标：平均跟踪奖励
                # 只有跟踪得好才 level up，跟踪差就 level down
                # 额外条件：必须存活足够长（避免刚开始就摔倒但误差低的情况）
                min_survival_for_up = 0.3  # 至少存活 30% 的 episode
                min_survival_for_eval = 0.1  # 至少存活 10% 才评估
                
                can_evaluate = survival_ratio >= min_survival_for_eval
                
                # Level up 条件：跟踪奖励高 且 存活足够长
                up_condition = (avg_tracking_reward > 0.7) & (survival_ratio > min_survival_for_up) & can_evaluate
                move_up[no_goals] = up_condition[no_goals]
                
                # Level down 条件：跟踪奖励低 或 存活太短
                down_condition = ((avg_tracking_reward < 0.4) | (survival_ratio < min_survival_for_up)) & can_evaluate
                # 如果存活太短无法评估，也 level down
                down_condition = down_condition | (~can_evaluate)
                move_down[no_goals] = down_condition[no_goals]
        else:
            # 兜底：使用跟踪奖励
            step_count = self.episode_step_count[env_ids]
            step_count_safe = torch.clamp(step_count, min=1.0)
            avg_tracking_reward = self.episode_cmd_tracking_reward[env_ids] / step_count_safe
            survival_ratio = self.episode_length_buf[env_ids].float() / self.max_episode_length
            
            move_up = (avg_tracking_reward > 0.7) & (survival_ratio > 0.3)
            move_down = (avg_tracking_reward < 0.4) | (survival_ratio < 0.1)

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0)
        )

        # 更新相关状态
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        if hasattr(self, 'terrain') and hasattr(self.terrain, 'has_goals'):
            has_goals_np = self.terrain.has_goals
            levels = self.terrain_levels[env_ids].cpu().numpy().astype(int)
            types = self.terrain_types[env_ids].cpu().numpy().astype(int)
            has_goals_flat = has_goals_np[levels, types]
            self.env_has_goals[env_ids] = torch.from_numpy(has_goals_flat).to(self.device).to(torch.bool)
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        self._update_env_bounds(env_ids)

    def _update_command_curriculum(self, env_ids):
        """
        基于 terrain_levels 更新命令范围（分段curriculum）

        设计原则：
        - 初始范围（低难度）：命令从两个不包含0的分段区间开始采样
          负向：[max_neg * min_ratio, -dead_zone]
          正向：[dead_zone, max_pos * min_ratio]
        - 范围扩展：随难度增加，两个区间向两侧对称扩张
        - 最终范围：覆盖完整的 max_range，包括0附近
        """
        if not self.init_done:
            return
        terrain_levels = self.terrain_levels[env_ids]
        max_level = self.max_terrain_level - 1
        # 计算难度比例 [0, 1]
        level_ratio = torch.clamp(terrain_levels.float() / max(max_level, 1), 0.0, 1.0)
        # 获取配置参数
        min_ratio = self.cfg.commands.min_ratio  # 初始范围比例
        dead_zone = self.cfg.commands.initial_dead_zone  # 初始禁区半径
        # 遍历每个命令维度
        for cmd_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            max_range = self.command_max_ranges[cmd_name]
            max_neg = max_range[0]  # 负向最大值（如 -1.0）
            max_pos = max_range[1]  # 正向最大值（如 1.5）
            # === 计算分段范围 ===
            # 负向范围：从 [max_neg * min_ratio, -dead_zone] 扩展到 [max_neg, 0]
            neg_inner = -dead_zone[cmd_name]  # 负向内边界（固定）
            neg_outer_init = max_neg * min_ratio  # 负向外边界初始值
            neg_outer = neg_outer_init + level_ratio * (max_neg - neg_outer_init)
            # 正向范围：从 [dead_zone, max_pos * min_ratio] 扩展到 [0, max_pos]
            pos_inner = dead_zone[cmd_name]  # 正向内边界（固定）
            pos_outer_init = max_pos * min_ratio  # 正向外边界初始值
            pos_outer = pos_outer_init + level_ratio * (max_pos - pos_outer_init)
            # 内边界随难度收缩到0
            # 当 level_ratio=1 时，neg_inner→0, pos_inner→0
            neg_inner_current = -dead_zone[cmd_name] * (1.0 - level_ratio)
            pos_inner_current = dead_zone[cmd_name] * (1.0 - level_ratio)
            # 存储分段范围（后续采样时使用）
            # 格式：[neg_outer, neg_inner, pos_inner, pos_outer]
            # 例如：level_ratio=0 时 → [-0.3, -0.3, 0.3, 0.45]
            #       level_ratio=1 时 → [-1.0, 0, 0, 1.5]
            self.command_ranges[cmd_name][env_ids, 0] = neg_outer
            self.command_ranges[cmd_name][env_ids, 1] = pos_outer
            # 将内边界存储到额外的tensor（用于采样逻辑）
            if not hasattr(self, 'command_dead_zones'):
                self.command_dead_zones = {
                    "lin_vel_x": torch.zeros(self.num_envs, 2, device=self.device),
                    "lin_vel_y": torch.zeros(self.num_envs, 2, device=self.device),
                    "ang_vel_z": torch.zeros(self.num_envs, 2, device=self.device),
                }
            self.command_dead_zones[cmd_name][env_ids, 0] = neg_inner_current
            self.command_dead_zones[cmd_name][env_ids, 1] = pos_inner_current


    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
            
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6) # for feet only, see create_env()
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs, device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 4, device=self.device, dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contact_forces = torch.zeros_like(self.contact_forces)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_dof_pos_all[:] = self.default_dof_pos[0]

        self.height_update_interval = 1
        if hasattr(self.cfg.env, "height_update_dt"):
            self.height_update_interval = int(self.cfg.env.height_update_dt / (self.cfg.sim.dt * self.cfg.control.decimation))

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.buffer_len, 
                                            self.cfg.depth.resized[1], 
                                            self.cfg.depth.resized[0]).to(self.device)
            self.depth_buffer_clean = torch.zeros(self.num_envs,
                                                  self.cfg.depth.buffer_len,
                                                  self.cfg.depth.resized[1],
                                                  self.cfg.depth.resized[0]).to(self.device)

        # bounds check
        self.env_bounds = torch.zeros(self.num_envs, 4, device=self.device)  # [x_min, x_max, y_min, y_max]
        self._compute_env_bounds()

        # goals command
        self.env_has_goals = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if hasattr(self, 'terrain') and hasattr(self.terrain, 'has_goals'):
            has_goals_np = self.terrain.has_goals  # shape: (num_rows, num_cols)
            # 根据 terrain_levels 和 terrain_types 获取每个 env 是否有 goals
            if hasattr(self, 'terrain_levels') and hasattr(self, 'terrain_types'):
                levels = self.terrain_levels.cpu().numpy().astype(int)
                types = self.terrain_types.cpu().numpy().astype(int)
                has_goals_flat = has_goals_np[levels, types]  # numpy array (num_envs,)
                self.env_has_goals[:] = torch.from_numpy(has_goals_flat).to(self.device).to(torch.bool)
        
        # 用于存储随机采样的 x 命令（当 x_stop_by_yaw 时使用）
        self.sampled_x_cmd_buffer = torch.zeros(self.num_envs, device=self.device)
        # update terrain levels
        self.episode_traveled_distance = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.episode_max_possible_distance = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        # 新增：累积命令跟踪误差和累积奖励
        self.episode_cmd_tracking_error = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.episode_cmd_tracking_reward = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.episode_step_count = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        # 新增：用于记录线速度跟踪奖励的滑动平均
        self.lin_vel_tracking_reward_avg = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.ang_vel_enabled = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)  # 是否启用角速度命令

        # 初始化命令范围（per-env）
        self.command_ranges = {
            "lin_vel_x": torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device),
            "lin_vel_y": torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device),
            "ang_vel_z": torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device),
        }
        # 初始化禁区边界（per-env）
        self.command_dead_zones = {
            "lin_vel_x": torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device),
            "lin_vel_y": torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device),
            "ang_vel_z": torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device),
        }
        # 设置初始值为最低难度的分段范围
        dead_zone = {}
        for cmd_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            max_range = self.command_max_ranges[cmd_name]
            min_ratio = self.cfg.commands.min_ratio
            dead_zone[cmd_name] = self.cfg.commands.initial_dead_zone[cmd_name]
            # 外边界
            self.command_ranges[cmd_name][:, 0] = max_range[0] * min_ratio
            self.command_ranges[cmd_name][:, 1] = max_range[1] * min_ratio
            # 内边界（初始禁区）
            self.command_dead_zones[cmd_name][:, 0] = -dead_zone[cmd_name]
            self.command_dead_zones[cmd_name][:, 1] = dead_zone[cmd_name]
        # 初始采样
        self._resample_commands(torch.arange(self.num_envs, device=self.device, requires_grad=False))

    def _compute_env_bounds(self):
        """
        计算每个环境的逻辑边界（世界坐标）
        边界 = subterrain 的物理范围，与 agent 初始位置无关
        """
        border_margin = getattr(self.cfg.env, 'border_margin', 0.1)
        
        env_length = self.cfg.terrain.terrain_length
        env_width = self.cfg.terrain.terrain_width
        
        # 直接基于 terrain_levels (row i) 和 terrain_types (col j) 计算
        # subterrain 范围：
        #   x: [i * env_length, (i+1) * env_length]
        #   y: [j * env_width, (j+1) * env_width]
        
        # terrain_levels 对应 row index (i)
        # terrain_types 对应 col index (j)
        row_indices = self.terrain_levels.float()  # i
        col_indices = self.terrain_types.float()   # j
        
        # x 方向边界
        self.env_bounds[:, 0] = row_indices * env_length + border_margin  # x_min
        self.env_bounds[:, 1] = (row_indices + 1) * env_length - border_margin  # x_max
        
        # y 方向边界
        self.env_bounds[:, 2] = col_indices * env_width + border_margin  # y_min
        self.env_bounds[:, 3] = (col_indices + 1) * env_width - border_margin  # y_max

    def _update_env_bounds(self, env_ids):
        """
        当环境的 terrain level/type 改变时，更新对应的边界
        """
        border_margin = getattr(self.cfg.env, 'border_margin', 0.1)
        env_length = self.cfg.terrain.terrain_length
        env_width = self.cfg.terrain.terrain_width

        row_indices = self.terrain_levels[env_ids].float()
        col_indices = self.terrain_types[env_ids].float()
        
        self.env_bounds[env_ids, 0] = row_indices * env_length + border_margin
        self.env_bounds[env_ids, 1] = (row_indices + 1) * env_length - border_margin
        self.env_bounds[env_ids, 2] = col_indices * env_width + border_margin
        self.env_bounds[env_ids, 3] = (col_indices + 1) * env_width - border_margin

    def _check_out_of_bounds(self):
        """
        检测 agent 是否超出 subterrain 边界
        Returns:
            out_of_bounds: (num_envs,) bool tensor，True 表示越界
        """
        pos_x = self.root_states[:, 0]
        pos_y = self.root_states[:, 1]
        
        out_of_bounds = (
            (pos_x < self.env_bounds[:, 0]) |
            (pos_x > self.env_bounds[:, 1]) |
            (pos_y < self.env_bounds[:, 2]) |
            (pos_y > self.env_bounds[:, 3])
        )
        return out_of_bounds

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border
        hf_params.transform.p.y = -self.terrain.border
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.flatten(order='C'), hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = config.original[0]
            camera_props.height = config.original[1]
            camera_props.enable_tensors = True

            if hasattr(config, "near_plane"):
                camera_props.near_plane = config.near_plane # meters
                print('Near plane has been set: ', camera_props.near_plane)

            camera_horizontal_fov = config.horizontal_fov 
            if isinstance(camera_horizontal_fov, (tuple, list)):
                # fov randomization
                camera_props.horizontal_fov = np.random.uniform(
                    camera_horizontal_fov[0], camera_horizontal_fov[1])
                print('Camera horizontal fov has been randomized: ', camera_horizontal_fov)
            else:
                camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)
            
            local_transform = gymapi.Transform()
            
            if isinstance(config.position, dict):
                cam_x = np.random.normal(config.position['mean'][0], config.position['std'][0])
                cam_y = np.random.normal(config.position['mean'][1], config.position['std'][1])
                cam_z = np.random.normal(config.position['mean'][2], config.position['std'][2])
                local_transform.p = gymapi.Vec3(cam_x, cam_y, cam_z)
                print('Camera position: ', cam_x, cam_y, cam_z)
                print('Camera position has been randomized: ', config.position)
            else:
                camera_position = np.copy(config.position)
                local_transform.p = gymapi.Vec3(*camera_position)

            if hasattr(config, "rotation"):
                if isinstance(config.rotation, dict):
                    cam_roll = np.random.uniform(0, 1) * (
                        config.rotation["upper"][0] - config.rotation["lower"][0]) + config.rotation["lower"][0]
                    cam_pitch = np.random.uniform(0, 1) * (
                        config.rotation["upper"][1] - config.rotation["lower"][1]) + config.rotation["lower"][1]
                    cam_yaw = np.random.uniform(0, 1) * (
                        config.rotation["upper"][2] - config.rotation["lower"][2]) + config.rotation["lower"][2]
                    local_transform.r = gymapi.Quat.from_euler_zyx(cam_roll, cam_pitch, cam_yaw)
                    print('Camera rotation: ', cam_roll, cam_pitch, cam_yaw)
                    print('Camera rotation has been randomized: ', config.rotation)
            else:
                camera_angle = np.random.uniform(config.angle[0], config.angle[1])
                local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)

            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
            
            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]


        for s in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.env.rand_yaw_range*np.random.uniform(-1, 1))
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
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

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        thigh_names = ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(thigh_names):
            self.thigh_indices[i] = self.dof_names.index(name)
        calf_names = ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(calf_names):
            self.calf_indices[i] = self.dof_names.index(name)
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level if self.cfg.terrain.max_init_terrain_level <= (self.cfg.terrain.num_rows - 1) else (self.cfg.terrain.num_rows - 1)
            if not self.cfg.terrain.curriculum: 
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]

            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
            self.cur_goals = self._gather_cur_goals()
            self.next_goals = self._gather_cur_goals(future=1)

        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        reward_norm_factor = 1#np.sum(list(self.reward_scales.values()))
        for rew in self.reward_scales:
            self.reward_scales[rew] = self.reward_scales[rew] / reward_norm_factor
        self.command_max_ranges = class_to_dict(self.cfg.commands.max_ranges)
        self.command_ranges = None
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_height_samples(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        i = self.lookat_id
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.measured_heights[i].cpu().numpy()
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
    def _draw_reconstructed_height_samples(self, recon_heights=None, env_id=None):
        """ Draw reconstructed height samples.
            recon_heights: tensor (num_envs, num_height_points) or (num_height_points,)
            If recon_heights is None, try to read self.extras["recon_heights"] or self.recon_heights.
        """
        if not self.terrain.cfg.measure_heights:
            return
        # try to obtain recon_heights from extras/attr when not provided
        if recon_heights is None:
            try:
                recon_heights = self.extras.get("recon_heights", None)
            except Exception:
                recon_heights = getattr(self, "recon_heights", None)
            if recon_heights is None:
                return
        # ensure tensor on cpu and torch.Tensor type
        if isinstance(recon_heights, np.ndarray):
            recon_heights = torch.from_numpy(recon_heights)
        if isinstance(recon_heights, torch.Tensor):
            recon_heights = recon_heights.cpu()
        else:
            return
        # ensure tensors refreshed so base_quat/root_states are up-to-date
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 1))  # cyan for recon
        i = self.lookat_id if env_id is None else env_id
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        # determine per-env heights
        # recon_heights shape can be (num_envs, num_points) or (num_points,)
        if recon_heights.dim() == 1 or recon_heights.shape[0] == self.num_height_points:
            heights = recon_heights.numpy()
        else:
            # recon_heights[i] -> (num_points,)
            heights = recon_heights[i].numpy()
        # same sampling points as original drawing
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = float(heights[j])
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_goals(self):
        if hasattr(self, "env_has_goals") and not self.env_has_goals[self.lookat_id]:
            return
        sphere_geom_passed = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 1, 0))  # 已经过的 goal: 绿色
        sphere_geom_cur = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 0, 1))     # 当前目标: 蓝色
        sphere_geom_next = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))    # 下一个目标: 红色
        sphere_geom_default = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0.6, 0.6, 0.6))  # 其他: 灰色
        sphere_geom_reached = gymutil.WireframeSphereGeometry(self.cfg.env.next_goal_threshold, 32, 32, None, color=(0, 1, 0))
        goals = self.terrain_goals[self.terrain_levels[self.lookat_id], self.terrain_types[self.lookat_id]].cpu().numpy()
        cur_idx = int(self.cur_goal_idx[self.lookat_id].cpu().item())
        n_goals = goals.shape[0]
        for i, goal in enumerate(goals):
            goal_xy = goal[:2] + self.terrain.cfg.border_size
            pts = (goal_xy/self.terrain.cfg.horizontal_scale).astype(int)
            goal_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale
            pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
            if i < cur_idx:
                gymutil.draw_lines(sphere_geom_passed, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            elif i == cur_idx:
                gymutil.draw_lines(sphere_geom_cur, self.gym, self.viewer, self.envs[self.lookat_id], pose)
                if self.reached_goal_ids[self.lookat_id]:
                    gymutil.draw_lines(sphere_geom_reached, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            elif i == cur_idx + 1 and (cur_idx + 1) < n_goals:
                gymutil.draw_lines(sphere_geom_next, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            else:
                gymutil.draw_lines(sphere_geom_default, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
        if not self.cfg.depth.use_camera:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
            pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
            for i in range(5):
                norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))
            for i in range(5):
                norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.2*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
    def _draw_feet(self):
        if hasattr(self, 'feet_at_edge'):
            non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))
            edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0, 0))

            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            n_feet = self.feet_indices.shape[0]
            env_handle = self.envs[self.lookat_id]
            for fi in range(n_feet):
                pose = gymapi.Transform(gymapi.Vec3(feet_pos[self.lookat_id, fi, 0], 
                                                    feet_pos[self.lookat_id, fi, 1], 
                                                    feet_pos[self.lookat_id, fi, 2]), r=None)
                if self.feet_at_edge[self.lookat_id, fi]:
                    gymutil.draw_lines(edge_geom, self.gym, self.viewer, env_handle, pose)
                else:
                    gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, env_handle, pose)

    def _draw_commands(self, env_id=None):
        """
        Visualize commands for one env:
          - linear command (vx, vy) as arrow (shaft + head)
          - angular command wz as an arc around the robot (signed arc angle)
        Default: draw only for lookat_id (avoid drawing all envs for perf).
        """
        if not hasattr(self, "commands"):
            return
        if env_id is None:
            env_id = self.lookat_id
        try:
            i = int(env_id)
        except Exception:
            i = int(self.lookat_id)
        env_handle = self.envs[i]

        # get base pose and commands (CPU/numpy)
        base_pos = self.root_states[i, :3].cpu().numpy()
        yaw = float(self.yaw[i].cpu().item())
        vx = float(self.commands[i, 0].cpu().item())
        vy = float(self.commands[i, 1].cpu().item())
        wz = float(self.commands[i, 2].cpu().item())

        # visualization scaling (meters per m/s for the arrow)
        lv_scale = getattr(self.cfg.viewer, "command_viz_lin_scale", 0.6)  # meters per (m/s)
        wz_scale = getattr(self.cfg.viewer, "command_viz_wz_scale", 1.0)   # radians per (rad/s) visualized arc angle
        arc_radius = getattr(self.cfg.viewer, "command_viz_arc_radius", 0.25)  # radius of arc for wz visualization

        # compute arrow tip in world frame (rotate by yaw)
        cos_y = np.cos(yaw); sin_y = np.sin(yaw)
        dx = vx * cos_y - vy * sin_y
        dy = vx * sin_y + vy * cos_y
        tip = base_pos + np.array([dx * lv_scale, dy * lv_scale, 0.0], dtype=np.float32)

        # build arrow vertices (shaft and two head points)
        shaft_p0 = np.array([base_pos[0], base_pos[1], base_pos[2] + 0.02], dtype=np.float32)
        shaft_p1 = np.array([tip[0], tip[1], tip[2] + 0.02], dtype=np.float32)

        # arrow head: two small lines
        dir_vec = np.array([shaft_p1[0] - shaft_p0[0], shaft_p1[1] - shaft_p0[1]], dtype=np.float32)
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-4:
            head_p1 = shaft_p1 + np.array([0.03, 0.0, 0.0], dtype=np.float32)
            head_p2 = shaft_p1 + np.array([-0.03, 0.0, 0.0], dtype=np.float32)
        else:
            head_dir = dir_vec / norm
            perp = np.array([-head_dir[1], head_dir[0]])
            head_size = lv_scale * 0.2
            head_p1 = shaft_p1 - np.concatenate([head_dir * head_size, [0.0]]) + np.concatenate([perp * head_size * 0.6, [0.0]])
            head_p2 = shaft_p1 - np.concatenate([head_dir * head_size, [0.0]]) - np.concatenate([perp * head_size * 0.6, [0.0]])

        arrow_points = np.stack([shaft_p0, shaft_p1, head_p1, head_p2], axis=0)
        arrow_indices = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int32)

        # color: blue if env has goal else green
        color = (0.0, 0.5, 1.0) if (hasattr(self, "env_has_goals") and self.env_has_goals[i]) else (0.0, 1.0, 0.0)

        # create simple line geometry
        class SimpleLineGeom:
            def __init__(self, points, indices, color):
                self.points = np.array(points, dtype=np.float32)
                self.indices = np.array(indices, dtype=np.int32)
                self._colors = np.tile(np.array(color, dtype=np.float32), (len(self.indices), 1))
            def num_lines(self):
                return len(self.indices)
            def colors(self):
                return self._colors
            def instance_verts(self, pose):
                # pose.p unused; points are absolute world coords
                verts = np.empty((len(self.indices) * 2, 3), dtype=np.float32)
                for idx, (a, b) in enumerate(self.indices):
                    verts[2*idx] = self.points[a]
                    verts[2*idx+1] = self.points[b]
                return verts

        # draw arrow (linear command)
        arrow_geom = SimpleLineGeom(arrow_points, arrow_indices, color)
        try:
            gymutil.draw_lines(arrow_geom, self.gym, self.viewer, env_handle, gymapi.Transform())
        except Exception:
            pass

        # draw angular command as arc (wz)
        # map wz to arc angle
        arc_angle = float(np.clip(wz * wz_scale, -np.pi, np.pi))
        if abs(arc_angle) > 1e-4:
            n_seg = 16
            # center on robot (slightly above base)
            center = base_pos.copy(); center[2] += 0.05
            # starting angle aligned with robot forward
            start_angle = yaw - arc_angle/2.0
            thetas = np.linspace(start_angle, start_angle + arc_angle, n_seg)
            arc_pts = []
            for th in thetas:
                x = center[0] + arc_radius * np.cos(th)
                y = center[1] + arc_radius * np.sin(th)
                arc_pts.append([x, y, center[2]])
            arc_pts = np.array(arc_pts, dtype=np.float32)
            arc_indices = np.stack([np.arange(0, n_seg-1), np.arange(1, n_seg)], axis=1).astype(np.int32)
            arc_color = (1.0, 0.4, 0.2) if arc_angle > 0 else (1.0, 0.2, 1.0)
            arc_geom = SimpleLineGeom(arc_pts, arc_indices, arc_color)
            try:
                gymutil.draw_lines(arc_geom, self.gym, self.viewer, env_handle, gymapi.Transform())
            except Exception:
                pass

    def _draw_env_bounds(self, env_id=None):
        """
        可视化逻辑边界：使用真实线段绘制（通过 draw_lines helper -> gym.add_lines）
        env_id: int or None -> 若为 None 使用 lookat_id
        """
        if not hasattr(self, "env_bounds") or self.env_bounds is None:
            return
        if env_id is None:
            env_id = self.lookat_id
        try:
            b = self.env_bounds[env_id].cpu().numpy()
            x_min, x_max, y_min, y_max = b
        except Exception:
            return
        # 在地面略微抬高 z 以便能看到
        z = float(self.env_origins[env_id, 2].cpu().item()) + 0.05
        env_handle = self.envs[env_id]
        # 四个角（按顺时针），使用绝对坐标（相对于 env）
        corners = np.array([
            [x_min, y_min, z],
            [x_max, y_min, z],
            [x_max, y_max, z],
            [x_min, y_max, z],
        ], dtype=np.float32)
        # 线段对（每行一个线段，索引对应 corners）
        line_indices = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
        # 简易 LineGeometry，满足 instance_verts(pose), num_lines(), colors()
        class SimpleLineGeom:
            def __init__(self, points, indices, color):
                self.points = np.array(points, dtype=np.float32)
                self.indices = np.array(indices, dtype=np.int32)
                self._colors = np.tile(np.array(color, dtype=np.float32), (len(self.indices), 1))
            def num_lines(self):
                return len(self.indices)
            def colors(self):
                return self._colors
            def instance_verts(self, pose):
                # pose 可能包含位移（通常我们传零位移），这里简单将 points + pose.p
                t = np.array([pose.p.x, pose.p.y, pose.p.z], dtype=np.float32)
                verts = np.empty((len(self.indices) * 2, 3), dtype=np.float32)
                for i, (a, b) in enumerate(self.indices):
                    verts[2 * i] = self.points[a] + t
                    verts[2 * i + 1] = self.points[b] + t
                return verts
        # 颜色：红色
        color = [1.0, 0.0, 0.0]
        geom = SimpleLineGeom(corners, line_indices, color)
        # 使用用户给出的 draw_lines helper（内部会调用 gym.add_lines）
        # pose 设为零平移（因为我们在 geom 中使用的是绝对坐标）
        pose = gymapi.Transform()
        try:
            gymutil.draw_lines(geom, self.gym, self.viewer, env_handle, pose)
        except Exception:
            num_points_per_edge = getattr(self.cfg.env, "env_bound_line_points", 20)
            point_radius = getattr(self.cfg.env, "env_bound_point_radius", 0.03)
            sphere_geom = gymutil.WireframeSphereGeometry(point_radius, 6, 6, None, color=(1, 1, 0))
            for i in range(4):
                a = corners[i]
                b = corners[(i + 1) % 4]
                for t in np.linspace(0.0, 1.0, num_points_per_edge):
                    p = a * (1.0 - t) + b * t
                    sphere_pose = gymapi.Transform(gymapi.Vec3(float(p[0]), float(p[1]), float(p[2])), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, env_handle, sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points
    
    def _init_foot_height_points(self):
        """
        Create polar sampling points around each foot.
        Returns foot_height_points of shape:
            (num_envs, num_feet=4, num_points_per_foot, 3)
        """
        radii = torch.tensor(self.cfg.terrain.foot_scan_radii, device=self.device)
        num_angles = self.cfg.terrain.foot_scan_num_angles
        angles = torch.linspace(0, 2 * torch.pi, num_angles, device=self.device, endpoint=False)

        # Polar → Cartesian
        # For each radius r and angle θ: (x=r*cosθ, y=r*sinθ)
        all_xy = []
        for r in radii:
            x = r * torch.cos(angles)
            y = r * torch.sin(angles)
            xy = torch.stack([x, y], dim=-1)  # (num_angles, 2)
            all_xy.append(xy)

        # concat radii: shape = (num_radii * num_angles, 2)
        all_xy = torch.cat(all_xy, dim=0)

        num_points = all_xy.shape[0]
        num_feet = self.num_feet  # e.g. 4 (FL, FR, RL, RR)

        # Build full tensor
        foot_height_points = torch.zeros(
            (self.num_envs, num_feet, num_points, 3),
            device=self.device,
            requires_grad=False
        )

        # Fill xy for each env and each foot
        foot_height_points[:, :, :, 0:2] = all_xy  # repeated automatically

        self.num_foot_height_points = num_points
        return foot_height_points


    def get_foot_contacts(self):
        foot_contacts_bool = self.contact_forces[:, self.feet_indices, 2] > 10
        if self.cfg.env.include_foot_contacts:
            return foot_contacts_bool
        else:
            return torch.zeros_like(foot_contacts_bool).to(self.device)

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
            (num_envs, num_height_points) or env_ids: (len(env_ids), num_height_points)
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_foot_heights(self, env_ids=None):
        """
        Sample height values at the polar grid around each foot.
        Returns shape:
            (num_envs, num_feet, num_points_per_foot)
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(
                (self.num_envs, self.num_feet, self.num_foot_height_points),
                device=self.device
            )

        # Select envs
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # (num_envs, num_feet, num_points, 3)
        rel_points = self.foot_height_points[env_ids]  

        # Add each foot's world position
        # foot_positions: (num_envs, num_feet, 3)
        foot_world = self.foot_positions[env_ids].unsqueeze(2)  # → (env, feet, 1, 3)
        points = rel_points + foot_world

        # If you want to rotate based on foot yaw or base yaw (可选)
        # Example using base yaw:
        points = quat_apply_yaw(
            self.base_quat[env_ids].unsqueeze(1).unsqueeze(1).repeat(1, self.num_feet, self.num_foot_height_points, 1),
            points
        )

        # Add border
        points += self.terrain.cfg.border_size

        # Project into heightmap index
        px = (points[:, :, :, 0] / self.terrain.cfg.horizontal_scale).long()
        py = (points[:, :, :, 1] / self.terrain.cfg.horizontal_scale).long()

        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        # Bilinear-like min sampling (matching your original code)
        h1 = self.height_samples[px, py]
        h2 = self.height_samples[px + 1, py]
        h3 = self.height_samples[px, py + 1]

        h = torch.min(torch.min(h1, h2), h3)  # (env, feet, points)

        return h * self.terrain.cfg.vertical_scale


    def _get_heights_points(self, coords, env_ids=None):
        if env_ids:
            points = coords[env_ids]
        else:
            points = coords

        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    ################## parkour rewards ##################

    # def _reward_tracking_goal_vel(self):
    #     norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
    #     target_vec_norm = self.target_pos_rel / (norm + 1e-5)
    #     cur_vel = self.root_states[:, 7:9]
    #     rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
    #     return rew
    
    # def _reward_tracking_yaw(self):
    #     rew = torch.exp(-torch.abs(wrap_to_pi(self.target_yaw - self.yaw)))
    #     return rew
        
    # def _reward_tracking_lin_vel_x(self):
    #     cur_lin_vel_x = self.base_lin_vel[:, 0]
    #     target_lin_vel_x = self.commands[:, 0]
    #     error = cur_lin_vel_x - target_lin_vel_x
    #     rew = torch.exp(-torch.square(error))
    #     stop_mask = (torch.abs(target_lin_vel_x) < 0.1)
    #     rew[stop_mask] *= 2.0
    #     return rew

    def _reward_tracking_lin_vel_forward(self):
        """追踪前向速度（vx > 0）"""
        forward_mask = self.commands[:, 0] > 0
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        rew = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return rew * forward_mask.float()

    def _reward_tracking_lin_vel_backward(self):
        """追踪后向速度（vx < 0）"""
        backward_mask = self.commands[:, 0] < 0
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        rew = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return rew * backward_mask.float()

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel_z(self):
        """
        跟踪角速度命令 wz
        """
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        rew[self.env_class != 17] *= 0.5
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
     
    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        # rew[self.env_class != 17] = 0.
        return rew
    
    def _reward_roll_orientation(self):
        roll_component = self.projected_gravity[:, 0]
        rew = torch.square(roll_component)
        return rew
    
    def _reward_pitch_orientation(self):
        pitch_component = self.projected_gravity[:, 1]
        rew = torch.square(pitch_component)
        return rew

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    
    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    

    # def _reward_feet_phase(self):
    #     """
    #     纯相位比例奖励：
    #     - 不直接奖励 air_time
    #     - 只约束 stance / (stance + swing) 比例
    #     - 在落地事件结算
    #     """
    #     num_feet = self.feet_indices.shape[0]
    #     # ========= 接触判定 =========
    #     contact_z_curr = self.contact_forces[:, self.feet_indices, 2] > 1.0
    #     contact_z_last = self.last_contact_forces[:, self.feet_indices, 2] > 1.0
    #     contact_filt = torch.logical_or(contact_z_curr, contact_z_last)
    #     contact_support = torch.norm(
    #         self.contact_forces[:, self.feet_indices], dim=-1
    #     ) > 2.0
    #     # ========= 初始化 =========
    #     if not hasattr(self, "feet_air_time"):
    #         self.feet_air_time = torch.zeros(self.num_envs, num_feet, device=self.device)
    #     if not hasattr(self, "feet_stance_time"):
    #         self.feet_stance_time = torch.zeros(self.num_envs, num_feet, device=self.device)
    #     if not hasattr(self, "last_stance_duration"):
    #         self.last_stance_duration = torch.zeros(self.num_envs, num_feet, device=self.device)
    #     # ========= 事件 =========
    #     first_contact = (self.feet_air_time > 0.0) & contact_filt
    #     if not hasattr(self, "_prev_contact_support"):
    #         self._prev_contact_support = contact_support.clone()
    #     lift_off = self._prev_contact_support & (~contact_support)
    #     # ========= 计时 =========
    #     self.feet_air_time += self.dt
    #     self.feet_stance_time[contact_support] += self.dt
    #     self.last_stance_duration[lift_off] = self.feet_stance_time[lift_off]
    #     self.feet_stance_time[~contact_support] = 0.0
    #     # ========= 相位比例奖励 =========
    #     swing = self.feet_air_time
    #     stance = self.last_stance_duration
    #     cycle = stance + swing + 1e-6
    #     stance_ratio = stance / cycle
    #     desired_ratio = getattr(self.cfg.rewards, "desired_stance_ratio", 0.5)
    #     ratio_error = torch.abs(stance_ratio - desired_ratio)
    #     reward = torch.sum(ratio_error * first_contact.float(), dim=1)
    #     # ========= 命令门控 =========
    #     lin_vel_clip = getattr(self.cfg.commands, "lin_vel_clip", 0.1)
    #     ang_vel_clip = getattr(self.cfg.commands, "ang_vel_clip", 0.1)
    #     cmd_nonzero = torch.logical_or(
    #         torch.norm(self.commands[:, :2], dim=1) > lin_vel_clip,
    #         torch.abs(self.commands[:, 2]) > ang_vel_clip
    #     )
    #     reward *= cmd_nonzero.float()
    #     # ========= reset =========
    #     self.feet_air_time *= (~contact_filt).float()
    #     self._prev_contact_support = contact_support.clone()
    #     return reward
    
    def _reward_feet_phase(self):
        """
        相位比例 + 周期软下界奖励
        - desired_ratio 根据命令速度动态调整
        """
        num_feet = self.feet_indices.shape[0]
        # ========= 接触判定 =========
        contact_z_curr = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_z_last = self.last_contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact_z_curr, contact_z_last)
        contact_support = (
            torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.0
        )
        # ========= 初始化 =========
        if not hasattr(self, "feet_air_time"):
            self.feet_air_time = torch.zeros(self.num_envs, num_feet, device=self.device)
        if not hasattr(self, "feet_stance_time"):
            self.feet_stance_time = torch.zeros(self.num_envs, num_feet, device=self.device)
        if not hasattr(self, "last_stance_duration"):
            self.last_stance_duration = torch.zeros(self.num_envs, num_feet, device=self.device)
        # ========= 事件 =========
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        if not hasattr(self, "_prev_contact_support"):
            self._prev_contact_support = contact_support.clone()
        lift_off = self._prev_contact_support & (~contact_support)
        # ========= 计时 =========
        self.feet_air_time += self.dt
        self.feet_stance_time[contact_support] += self.dt
        self.last_stance_duration[lift_off] = self.feet_stance_time[lift_off]
        self.feet_stance_time[~contact_support] = 0.0
        # ========= 动态 desired_ratio（关键修改）=========
        # 计算命令速度的幅值（线速度 + 角速度）
        lin_vel_cmd = torch.norm(self.commands[:, :2], dim=1)  # (num_envs,)
        ang_vel_cmd = torch.abs(self.commands[:, 2])  # (num_envs,)
        # 综合速度指标（可以调整权重）
        cmd_speed = lin_vel_cmd + 0.5 * ang_vel_cmd  # (num_envs,)
        # 根据速度映射 desired_ratio
        # 配置参数
        ratio_at_zero_speed = getattr(self.cfg.rewards, "stance_ratio_at_low_speed", 0.7)  # 低速时的目标比例
        ratio_at_max_speed = getattr(self.cfg.rewards, "stance_ratio_at_high_speed", 0.4)   # 最大速度时的目标比例
        max_speed_for_ratio = getattr(self.cfg.commands.max_ranges, "lin_vel_x", [-1.0, 1.5])[1]  # 用于映射的最大速度
        min_speed_for_ratio = getattr(self.cfg.commands, "lin_vel_clip", 0.1)
        # 线性插值：speed=0 → ratio_at_zero_speed, speed=max_speed → ratio_at_max_speed
        # ratio = ratio_at_zero - (ratio_at_zero - ratio_at_max) * min(speed/max_speed, 1.0)
        cmd_speed_clipped = torch.clamp(cmd_speed, min_speed_for_ratio, max_speed_for_ratio)
        speed_normalized = (cmd_speed_clipped - min_speed_for_ratio) / (max_speed_for_ratio - min_speed_for_ratio)
        # sigmoid: 0.5 + 0.5 * tanh(x) 范围 [0, 1]
        speed_factor = torch.sigmoid(2.0 * (speed_normalized - 0.5))  # 在 0.5 处中心化
        desired_ratio = ratio_at_zero_speed - (ratio_at_zero_speed - ratio_at_max_speed) * speed_factor
        # desired_ratio shape: (num_envs,)
        # ========= 相位比例 =========
        swing = self.feet_air_time  # (num_envs, num_feet)
        stance = self.last_stance_duration  # (num_envs, num_feet)
        cycle = stance + swing + 1e-6
        # 扩展 desired_ratio 到每只脚
        desired_ratio_per_foot = desired_ratio[:, None].expand(-1, num_feet)  # (num_envs, num_feet)
        ratio_error = torch.abs(stance / cycle - desired_ratio_per_foot)
        # ========= 周期软下界 =========
        min_cycle_time = getattr(self.cfg.rewards, "min_cycle_time", 0.5)
        cycle_penalty = torch.clamp(min_cycle_time - cycle, min=0.0)
        # ========= 合成奖励（注意：这是惩罚项） =========
        reward = torch.sum(
            (ratio_error + cycle_penalty) * first_contact.float(),
            dim=1
        )
        # ========= 命令门控 =========
        lin_vel_clip = getattr(self.cfg.commands, "lin_vel_clip", 0.1)
        ang_vel_clip = getattr(self.cfg.commands, "ang_vel_clip", 0.1)
        cmd_nonzero = torch.logical_or(
            torch.norm(self.commands[:, :2], dim=1) > lin_vel_clip,
            torch.abs(self.commands[:, 2]) > ang_vel_clip
        )
        reward *= cmd_nonzero.float()
        # ========= reset =========
        self.feet_air_time *= (~contact_filt).float()
        self._prev_contact_support = contact_support.clone()
        return reward

    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     last_contact = self.last_contact_forces[:, self.feet_indices, 2] > 1.
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, last_contact) 
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    #     # 修改：线速度或角速度命令非零时都给予步态奖励
    #     lin_vel_clip = getattr(self.cfg.commands, 'lin_vel_clip', 0.1)
    #     ang_vel_clip = getattr(self.cfg.commands, 'ang_vel_clip', 0.1)
    #     cmd_nonzero = torch.logical_or(
    #         torch.norm(self.commands[:, :2], dim=1) > lin_vel_clip,
    #         torch.abs(self.commands[:, 2]) > ang_vel_clip
    #     )
    #     rew_airTime *= cmd_nonzero.float()
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime

    # def _reward_feet_min_contact_time(self):
    #     """
    #     惩罚接地时间过短（防止快速点地作弊）
    #     使用 contact_buf 的上一帧作为 prev_contact（因为 post_physics_step 在 rewards 前会更新 contact_buf）
    #     """
    #     # 当前接触：与 contact_buf 的计算方式保持一致（使用力的 norm）
    #     contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.0  # (num_envs, 4)
    #     # 上一帧接触：优先从 contact_buf 读取（compute_observations 会在 rewards 之后更新 contact_buf）
    #     if hasattr(self, "contact_buf") and self.contact_buf.shape[1] >= 1:
    #         prev_contact = (self.contact_buf[:, -1, :] > 0.5)  # contact_buf 保存的是 contact_filt 的历史，-1 是上一帧
    #     else:
    #         # fallback: 使用 last_contacts（如果没有 contact_buf）
    #         prev_contact = self.last_contacts
    #     # 初始化 feet_contact_time（每次接地累计）
    #     if not hasattr(self, 'feet_contact_time'):
    #         self.feet_contact_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], device=self.device)
    #     # 增加接地时间：仅对当前接触的脚计时
    #     self.feet_contact_time[contact] += self.dt
    #     # 检测离地事件（上一帧接触、当前不接触）
    #     lift_off = prev_contact & (~contact)
    #     # 阈值与惩罚
    #     min_contact_time = getattr(self.cfg.rewards, 'min_feet_contact_time', 0.1)
    #     # 计算接地时间过短的惩罚（只有在 lift_off 时生效）
    #     contact_too_short = torch.clamp(min_contact_time - self.feet_contact_time, min=0.0)
    #     penalty = torch.sum(contact_too_short * lift_off.float(), dim=1)
    #     # 重置已离地脚的接地计时（或未接触的脚）
    #     self.feet_contact_time[~contact] = 0.0
    #     # note: 返回的值会乘以 reward scale（cfg.rewards.scales.feet_min_contact_time）
    #     return penalty

    def _reward_feet_contact_balance(self):
        """
        惩罚单脚长时间悬空 - 鼓励四脚接触时间平衡
        """
        # 统计每只脚在 contact_buf 中的接触比例
        contact_ratio = self.contact_buf.mean(dim=1)  # (num_envs, 4)
        # 计算四脚接触率的方差（方差越小越平衡）
        variance = torch.var(contact_ratio, dim=1)
        # 返回负方差作为惩罚（方差越大惩罚越大）
        return variance

    # def _reward_gait_periodicity(self):
    #     """
    #     鼓励步态周期性 - 基于接触状态变化频率
    #     """
    #     if self.contact_buf.shape[1] < 2:
    #         return torch.zeros(self.num_envs, device=self.device)
    #     # 计算接触状态变化次数（从接触到悬空或反之）
    #     contact_changes = torch.abs(self.contact_buf[:, 1:] - self.contact_buf[:, :-1])
    #     change_freq = contact_changes.sum(dim=(1, 2))  # 总变化次数
    #     # 期望的变化频率（取决于 buffer 长度和期望步频）
    #     expected_changes = self.contact_buf.shape[1] * 0.3  # 经验值
    #     # 奖励接近期望频率的步态
    #     return torch.exp(-torch.abs(change_freq - expected_changes) / expected_changes)

    def _reward_lazy_stop(self):
        """
        惩罚：当命令非零时，根据实际速度与命令的误差进行惩罚
        使用 square：大误差时给予更强的纠正信号
        """
        lin_vel_clip = getattr(self.cfg.commands, 'lin_vel_clip', 0.1)
        ang_vel_clip = getattr(self.cfg.commands, 'ang_vel_clip', 0.1)

        lin_vel_error = torch.norm(self.base_lin_vel[:, :2] - self.commands[:, :2], dim=1)
        ang_vel_error = torch.abs(self.base_ang_vel[:, 2] - self.commands[:, 2])

        lin_cmd_nonzero = torch.norm(self.commands[:, :2], dim=1) > lin_vel_clip
        ang_cmd_nonzero = torch.abs(self.commands[:, 2]) > ang_vel_clip

        # 使用 square: 大误差时惩罚更强，有利于快速纠正
        lin_penalty = torch.square(lin_vel_error) * lin_cmd_nonzero.float()
        ang_penalty = torch.square(ang_vel_error) * ang_cmd_nonzero.float()

        return lin_penalty + ang_penalty
    
    def _reward_stand_still(self):
        """
        惩罚：当命令为零时机器人仍在运动或未保持稳定站立
        - 关节位置偏离默认值
        - 四脚未全部支撑
        """
        lin_vel_clip = getattr(self.cfg.commands, 'lin_vel_clip', 0.1)
        ang_vel_clip = getattr(self.cfg.commands, 'ang_vel_clip', 0.1)
        # 命令接近零
        cmd_near_zero = torch.logical_and(
            torch.norm(self.commands[:, :2], dim=1) < lin_vel_clip,
            torch.abs(self.commands[:, 2]) < ang_vel_clip
        )
        # 惩罚1: 关节位置偏离默认值
        joint_penalty = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        # 惩罚2: 四脚未全部支撑
        # 检测每只脚的支撑状态（基于力的范数）
        contact_support = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.0  # (num_envs, 4)
        num_feet_in_support = contact_support.sum(dim=1)  # 每个 env 有几只脚支撑
        # 当命令为零时，期望四脚全部支撑（num_feet_in_support == 4）
        # 惩罚 = (4 - 实际支撑脚数)，未支撑的脚越多惩罚越大
        feet_support_penalty = (4.0 - num_feet_in_support.float())
        # 合并两项惩罚（可根据需要调整权重）
        # 这里假设关节偏离和脚支撑同等重要
        total_penalty = (joint_penalty + feet_support_penalty) * cmd_near_zero.float()
        return total_penalty
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)