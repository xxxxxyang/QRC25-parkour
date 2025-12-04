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
import numpy as np


class Go2BaseCfg( LeggedRobotCfg ):
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
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_goal_vel = 0
            tracking_lin_vel = 1.5
            stand_still = -0.2
            orientation = -1.5
            feet_air_time = 1.0
            lazy_stop = -1.0
            dof_error = -0.1
            base_height = -0.1

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

    class terrain( LeggedRobotCfg.terrain ):
        curriculum = False # No curriculum
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
                        "parkour_flat": 1.,
                        "parkour_step": 0.,
                        "parkour_gap": 0.,
                        "demo": 0.,}
        terrain_proportions = list(terrain_dict.values())

    class commands( LeggedRobotCfg.commands ):
        heading_command = True
        curriculum = False
        class max_ranges( LeggedRobotCfg.commands.max_ranges ):
            # lin_vel_x = [0.3, 0.8]  # [m/s]
            lin_vel_x = [-1.0, 1.5]  # [m/s]
            lin_vel_y = [0.0, 0.0]  # [m/s]
            ang_yaw = [-np.pi, np.pi]  # [rad]
            delta_yaw_threhold = 0.5 # [rad] clip  (ang_yaw/command_yaw - cur_yaw) to this range

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.3, 2.0]

class Go2BaseCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'base'

