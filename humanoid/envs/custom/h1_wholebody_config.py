# SPDX-License-Identifier: BSD-3-Clause
from humanoid.envs.custom.h1_config import H1Cfg, H1CfgPPO


class H1WholeBodyCfg(H1Cfg):
    """H1 whole-body config — 18 DOF (10 legs + 8 arms).

    Arm joints appended after leg joints:
      10 left_shoulder_pitch   14 right_shoulder_pitch
      11 left_shoulder_roll    15 right_shoulder_roll
      12 left_shoulder_yaw     16 right_shoulder_yaw
      13 left_elbow            17 right_elbow
    """

    class env(H1Cfg.env):
        num_single_obs = 65      # 5 + 18 + 18 + 18 + 3 + 3
        num_observations = int(H1Cfg.env.frame_stack * num_single_obs)
        single_num_privileged_obs = 97   # 65 + 18(diff) + 3(lin_vel) + 2+3+1+1+2+2
        num_privileged_obs = int(H1Cfg.env.c_frame_stack * single_num_privileged_obs)
        num_actions = 18

    class asset(H1Cfg.asset):
        # same URDF but we now control arm joints too
        terminate_after_contacts_on = ['pelvis']
        penalize_contacts_on = ['pelvis']

    class init_state(H1Cfg.init_state):
        default_joint_angles = {
            # legs
            'left_hip_yaw_joint':    0.,
            'left_hip_roll_joint':   0.,
            'left_hip_pitch_joint':  -0.1,
            'left_knee_joint':        0.3,
            'left_ankle_joint':      -0.2,
            'right_hip_yaw_joint':   0.,
            'right_hip_roll_joint':  0.,
            'right_hip_pitch_joint': -0.1,
            'right_knee_joint':       0.3,
            'right_ankle_joint':     -0.2,
            # arms — neutral hang position
            'left_shoulder_pitch_joint':  0.3,
            'left_shoulder_roll_joint':   0.,
            'left_shoulder_yaw_joint':    0.,
            'left_elbow_joint':           0.9,
            'right_shoulder_pitch_joint': 0.3,
            'right_shoulder_roll_joint':  0.,
            'right_shoulder_yaw_joint':   0.,
            'right_elbow_joint':          0.9,
        }

    class control(H1Cfg.control):
        stiffness = {
            'hip_yaw': 200., 'hip_roll': 200., 'hip_pitch': 350.,
            'knee': 350., 'ankle': 15.,
            'shoulder_pitch': 80., 'shoulder_roll': 80., 'shoulder_yaw': 40.,
            'elbow': 60.,
        }
        damping = {
            'hip_yaw': 10., 'hip_roll': 10., 'hip_pitch': 10.,
            'knee': 10., 'ankle': 10.,
            'shoulder_pitch': 4., 'shoulder_roll': 4., 'shoulder_yaw': 2.,
            'elbow': 3.,
        }

    class rewards(H1Cfg.rewards):
        class scales(H1Cfg.rewards.scales):
            # encourage natural arm counter-swing during walking
            arm_swing = 0.3
            # keep arms close to neutral when not swinging
            arm_default_pos = 0.2


class H1WholeBodyCfgPPO(H1CfgPPO):

    class policy(H1CfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class runner(H1CfgPPO.runner):
        experiment_name = 'H1_wholebody_ppo'
        use_wandb = False
