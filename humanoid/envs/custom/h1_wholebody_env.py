# SPDX-License-Identifier: BSD-3-Clause
import torch
from humanoid.envs.custom.h1_env import H1FreeEnv


class H1WholeBodyEnv(H1FreeEnv):
    """H1 whole-body env — 18 DOF (legs + arms).

    Gait reference motion is computed only for the 10 leg joints (indices 0-9).
    Arm joints (indices 10-17) are encouraged to counter-swing with the gait
    and to stay near their default position.
    """

    def compute_ref_state(self):
        """Leg reference same as H1FreeEnv; arms get a counter-swing target."""
        super().compute_ref_state()

        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        swing_scale = self.cfg.rewards.target_joint_pos_scale * 0.5

        # left arm counter-swings with right leg, right arm with left leg
        self.ref_dof_pos[:, 10] = -self.ref_dof_pos[:, 7] * swing_scale  # L shoulder pitch
        self.ref_dof_pos[:, 14] = -self.ref_dof_pos[:, 2] * swing_scale  # R shoulder pitch
        self.ref_action = 2 * self.ref_dof_pos

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        ns = self.cfg.noise.noise_scales
        noise_vec[0:5]   = 0.
        noise_vec[5:23]  = ns.dof_pos * self.obs_scales.dof_pos   # 18 joints pos
        noise_vec[23:41] = ns.dof_vel * self.obs_scales.dof_vel   # 18 joints vel
        noise_vec[41:59] = 0.                                       # previous actions
        noise_vec[59:62] = ns.ang_vel * self.obs_scales.ang_vel
        noise_vec[62:65] = ns.quat * self.obs_scales.quat
        return noise_vec

    def _reward_arm_swing(self):
        """Reward counter-swing arm motion relative to opposite leg."""
        diff = self.dof_pos[:, 10:18] - self.ref_dof_pos[:, 10:18]
        return torch.exp(-2 * torch.norm(diff, dim=1))

    def _reward_arm_default_pos(self):
        """Keep arms near default neutral hang when not actively swinging."""
        arm_diff = self.dof_pos[:, 10:18] - self.default_dof_pos[:, 10:18]
        return torch.exp(-torch.norm(arm_diff, dim=1))
