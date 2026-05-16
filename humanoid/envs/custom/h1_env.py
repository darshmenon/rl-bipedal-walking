# SPDX-License-Identifier: BSD-3-Clause
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import torch
from humanoid.envs import LeggedRobot
from humanoid.utils.terrain import HumanoidTerrain
from humanoid.envs.custom.humanoid_env import XBotLFreeEnv


class H1FreeEnv(XBotLFreeEnv):
    """H1 leg-only (10-DOF) locomotion environment.

    Joint order expected from Isaac Gym URDF parsing:
      0  left_hip_yaw    5  right_hip_yaw
      1  left_hip_roll   6  right_hip_roll
      2  left_hip_pitch  7  right_hip_pitch
      3  left_knee       8  right_knee
      4  left_ankle      9  right_ankle
    """

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left swing phase
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1   # left_hip_pitch
        self.ref_dof_pos[:, 3] = sin_pos_l * scale_2   # left_knee
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1   # left_ankle
        # right swing phase
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 7] = sin_pos_r * scale_1   # right_hip_pitch
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_2   # right_knee
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_1   # right_ankle
        # double support
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        self.ref_action = 2 * self.ref_dof_pos

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        ns = self.cfg.noise.noise_scales
        noise_vec[0:5]   = 0.                                            # commands
        noise_vec[5:15]  = ns.dof_pos * self.obs_scales.dof_pos         # 10 joints pos
        noise_vec[15:25] = ns.dof_vel * self.obs_scales.dof_vel         # 10 joints vel
        noise_vec[25:35] = 0.                                            # previous actions
        noise_vec[35:38] = ns.ang_vel * self.obs_scales.ang_vel         # ang vel
        noise_vec[38:41] = ns.quat * self.obs_scales.quat               # euler
        return noise_vec
