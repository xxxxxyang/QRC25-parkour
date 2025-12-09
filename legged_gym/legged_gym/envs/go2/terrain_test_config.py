from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2TerrainTestCfg( LeggedRobotCfg ):
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
        add_terrain_border = True
        border_type = 'wall'  # 'wall' or 'pit'
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.25,
                        "rough slope down": 0.25,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0.25, 
                        "stepping stones": 0.,
                        "gaps": 0., 
                        "smooth flat": 0.25,
                        "pit": 0.,
                        "wall": 0.,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.,
                        "parkour_hurdle": 0.,
                        "parkour_flat": 0.,
                        "parkour_step": 0.,
                        "parkour_gap": 0.,
                        "demo": 0.,}
        terrain_proportions = list(terrain_dict.values())

class Go2TerrainTestCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'terrain_test'

  
