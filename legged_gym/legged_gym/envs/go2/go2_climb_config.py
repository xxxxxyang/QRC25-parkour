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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2ClimbCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        reset_warmup_steps = 10
        # num_envs = 6144

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1}     # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_new.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]#, "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        min_cycle_time = 0.7
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            # ===== 任务奖励 =====
            tracking_goal_vel = 2.5             # 主要任务奖励
            tracking_ang_vel_z = 0.3            # 辅助朝向对齐（delta_yaw->0）
            termination = -2.0

            tracking_lin_vel = 0.0              # 关掉
            tracking_lin_vel_forward = 0.0
            tracking_lin_vel_backward = 0.0

            # ===== 步态 =====
            feet_air_time = 1.
            feet_phase = 0.
            feet_contact_balance = 0.

            # ===== 姿态 =====
            base_height = -0.
            orientation = -1.0
            roll_orientation = -0.5
            lin_vel_z = -0.5                    # 惩罚上下的垂直速度

            # ===== 正则化 =====
            action_rate = -0.01
            torques = -2e-6
            delta_torques = -1e-7
            lazy_stop = -0.2                    # 保留，防止命令有速度时机器人不动
            dof_acc = -5e-8
            dof_error = -0.04
            dof_error_max = -0.05
            hip_pos = -1.0
            feet_stumble = -1.0
            feet_edge = -1.0

            # ===== 任务约束 =====
            stand_still = -1.0
            collision = -3.0

    class terrain( LeggedRobotCfg.terrain ):
        add_terrain_border = True
        border_type = 'wall'
        curriculum = True 
        # more rough
        downsampled_scale = 0.06
        height = [0.02, 0.12]
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.,
                        "rough slope down": 0.,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.,
                        "gaps": 0., 
                        "smooth flat": 0.,
                        "pit": 0.,
                        "wall": 0.,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.,
                        "parkour_hurdle": 0.,
                        "parkour_flat": 0.3,
                        "parkour_step": 0.4,
                        "parkour_gap": 0.3,
                        "demo": 0.,}
        terrain_proportions = list(terrain_dict.values())
        num_rows = 10
        num_cols = 20

    class commands(LeggedRobotCfg.commands):
        cmd_smooth_alpha = 0.85 # 平滑命令系数，越大越平滑
        curriculum = False
        # lin_vel_clip = 0.2 + 1e-7
        # ang_vel_clip = 0.2 + 1e-7      # 保留字段，不影响delta_yaw逻辑
        # min_ratio = 0.5

        # goal-based参数
        goal_vel_min  = 0.3        # parkour地形最小速度
        goal_vel_max  = 1.2        # parkour地形最大速度（也作no_goal地形上限）
        lin_vel_x_max = 1.5        # flat地形最大速度（双向）
        lin_vel_clip  = 0.1        # 零速dead zone

        class max_ranges(LeggedRobotCfg.commands.max_ranges):
            lin_vel_x = [-1.5, 1.8]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_z = [-1., 1.]       # 保留，lazy_stop clip用

    class depth( LeggedRobotCfg.depth ):
        # position = [0.32, 0.0, 0.035]  # front camera
        position = dict(
            mean = [0.32, 0.0, 0.035],
            std = [0.01, 0.01, 0.01]
        )
        rotation = dict(
            lower = [-0.1, -0.1, -0.1],
            upper = [0.1, 0.1, 0.1]
        )
        # angle = [-5.7, 5.7]  # positive pitch down

        horizontal_fov = [85, 89]
        near_plane = 0.1

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.3, 2.0]

class Go2ClimbCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.0

    class symmetry( LeggedRobotCfgPPO.symmetry ):
        enabled = True

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'climb'