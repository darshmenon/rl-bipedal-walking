# SPDX-License-Identifier: BSD-3-Clause
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1Cfg(LeggedRobotCfg):
    """Config for Unitree H1 — 10-DOF leg-only locomotion."""

    class env(LeggedRobotCfg.env):
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 41      # 5(cmd) + 10(pos) + 10(vel) + 10(act) + 3(ang_vel) + 3(euler)
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 65   # 41 + 10(diff) + 3(lin_vel) + 2(push_f) + 3(push_t) + 1(fric) + 1(mass) + 2(stance) + 2(contact)
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 10
        num_envs = 4096
        episode_length_s = 24
        use_ref_actions = False

    class safety:
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/humanoid_descriptions/ros/unitree_ros/robots/h1_description/urdf/h1.urdf'
        name = "H1"
        foot_name = "ankle_link"
        knee_name = "knee_link"
        terminate_after_contacts_on = ['pelvis']
        penalize_contacts_on = ['pelvis']
        self_collisions = 0
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20
        num_cols = 20
        max_init_terrain_level = 10
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.05]   # H1 stands ~1.0 m tall

        default_joint_angles = {
            'left_hip_yaw_joint':   0.,
            'left_hip_roll_joint':  0.,
            'left_hip_pitch_joint': -0.1,
            'left_knee_joint':       0.3,
            'left_ankle_joint':     -0.2,
            'right_hip_yaw_joint':  0.,
            'right_hip_roll_joint': 0.,
            'right_hip_pitch_joint':-0.1,
            'right_knee_joint':      0.3,
            'right_ankle_joint':    -0.2,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {
            'hip_yaw': 200., 'hip_roll': 200., 'hip_pitch': 350.,
            'knee': 350., 'ankle': 15.,
        }
        damping = {
            'hip_yaw': 10., 'hip_roll': 10., 'hip_pitch': 10.,
            'knee': 10., 'ankle': 10.,
        }
        action_scale = 0.25
        decimation = 10

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        up_axis = 1

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.1
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        action_delay = 0.5
        action_noise = 0.02

    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        resampling_time = 8.
        heading_command = True

        class ranges:
            lin_vel_x = [-0.3, 0.6]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.3, 0.3]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.98
        min_dist = 0.2
        max_dist = 0.5
        target_joint_pos_scale = 0.17
        target_feet_height = 0.06
        cycle_time = 0.64
        only_positive_rewards = True
        tracking_sigma = 5
        max_contact_force = 700

        class scales:
            joint_pos = 1.6
            feet_clearance = 1.
            feet_contact_number = 1.2
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            feet_contact_forces = -0.01
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5
            low_speed = 0.2
            track_vel_hard = 0.5
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class H1CfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60
        max_iterations = 3001
        save_interval = 100
        experiment_name = 'H1_ppo'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        use_wandb = False
