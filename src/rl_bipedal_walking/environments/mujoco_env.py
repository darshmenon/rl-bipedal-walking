"""
mujoco_env.py

Lightweight MuJoCo-based bipedal walking environment compatible with
Gymnasium and the SAC/PPO agents in this repo.  No ROS2 or Isaac Gym
required — only `mujoco` and `gymnasium`.

Supports XBot-L and Unitree H1 (leg-only) out of the box.
"""

import math
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation


_LEGGED_GYM_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)

ROBOT_CONFIGS = {
    'xbot': {
        'mjcf': os.path.join(_LEGGED_GYM_ROOT, 'resources', 'robots', 'XBot', 'mjcf', 'XBot-L.xml'),
        'num_actions': 12,
        'standing_height': 0.89,
        'kps': np.array([200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], np.float32),
        'kds': np.ones(12, np.float32) * 10,
        'tau_limit': np.ones(12, np.float32) * 200,
        'default_pos': np.zeros(12, np.float32),
        'use_imu_sensor': True,
    },
    'h1': {
        'mjcf': os.path.join(
            _LEGGED_GYM_ROOT, 'humanoid_descriptions', 'ros', 'unitree_ros',
            'robots', 'h1_description', 'mjcf', 'h1.xml'
        ),
        'num_actions': 10,
        'standing_height': 0.98,
        'kps': np.array([200, 200, 350, 350, 15, 200, 200, 350, 350, 15], np.float32),
        'kds': np.ones(10, np.float32) * 10,
        'tau_limit': np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40], np.float32),
        'default_pos': np.array([-0.0, 0., -0.1, 0.3, -0.2, -0.0, 0., -0.1, 0.3, -0.2], np.float32),
        'use_imu_sensor': False,
    },
}


class MujocoBipedalEnv(gym.Env):
    """MuJoCo bipedal walking env with a policy-compatible observation space.

    Observation (per step, no frame-stack):
      sin/cos phase (2) + cmd (3) + dof_pos (N) + dof_vel (N) + ang_vel (3) + euler (3)
      = 11 + 2N  floats

    Action: N joint position deltas, scaled by action_scale=0.25.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, robot='xbot', render_mode=None, max_episode_steps=2400,
                 cmd_vx=0.4, cmd_vy=0.0, cmd_yaw=0.0):
        super().__init__()

        try:
            import mujoco
        except ImportError:
            raise ImportError("pip install mujoco to use MujocoBipedalEnv")

        self._mujoco = mujoco
        cfg = ROBOT_CONFIGS[robot]
        self.robot = robot
        self.cfg = cfg
        self.n = cfg['num_actions']
        self.kps = cfg['kps']
        self.kds = cfg['kds']
        self.tau_limit = cfg['tau_limit']
        self.default_pos = cfg['default_pos'].copy()
        self.standing_height = cfg['standing_height']
        self.use_imu = cfg['use_imu_sensor']
        self.action_scale = 0.25
        self.dt = 0.001
        self.decimation = 10          # policy dt = 0.01 s
        self.max_steps = max_episode_steps
        self.cmd = np.array([cmd_vx, cmd_vy, cmd_yaw], np.float32)
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(cfg['mjcf'])
        self.model.opt.timestep = self.dt
        self.data  = mujoco.MjData(self.model)

        obs_dim = 11 + 2 * self.n
        high = np.full(obs_dim, np.inf, np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space      = spaces.Box(
            -1.0, 1.0, shape=(self.n,), dtype=np.float32)

        self._step = 0
        self._last_action = np.zeros(self.n, np.float32)
        self._viewer = None

    # ── gym interface ────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._mujoco.mj_resetData(self.model, self.data)
        # small random perturbation so the agent doesn't overfit to one start
        noise = self.np_random.uniform(-0.01, 0.01, self.n)
        self.data.qpos[7:7 + self.n] = self.default_pos + noise
        self._mujoco.mj_forward(self.model, self.data)
        self._step = 0
        self._last_action[:] = 0.
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_q = self.action_scale * action + self.default_pos

        for _ in range(self.decimation):
            q  = self.data.qpos[7:7 + self.n]
            dq = self.data.qvel[6:6 + self.n]
            tau = self.kps * (target_q - q) + self.kds * (0. - dq)
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)
            ctrl = np.zeros(self.model.nu, np.float32)
            ctrl[:self.n] = tau
            self.data.ctrl = ctrl
            self._mujoco.mj_step(self.model, self.data)

        self._step += 1
        self._last_action = action.copy()

        obs = self._obs()
        reward, reward_info = self._reward(action)
        terminated = self._is_terminated()
        truncated  = self._step >= self.max_steps

        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, reward_info

    def render(self):
        try:
            import mujoco_viewer
        except ImportError:
            return
        if self._viewer is None:
            self._viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self._viewer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ── internals ────────────────────────────────────────────────────────

    def _quat_to_euler(self):
        if self.use_imu:
            quat_xyzw = self.data.sensor('orientation').data[[1, 2, 3, 0]]
        else:
            quat_xyzw = self.data.qpos[3:7][[1, 2, 3, 0]]
        r = Rotation.from_quat(quat_xyzw)
        return r.as_euler('xyz').astype(np.float32)

    def _ang_vel(self):
        if self.use_imu:
            return self.data.sensor('angular-velocity').data.astype(np.float32)
        return self.data.sensor('imu-angular-velocity').data.astype(np.float32)

    def _obs(self):
        phase = self._step * self.dt * self.decimation / 0.64
        sin_p = math.sin(2 * math.pi * phase)
        cos_p = math.cos(2 * math.pi * phase)

        q  = (self.data.qpos[7:7 + self.n] - self.default_pos).astype(np.float32)
        dq = self.data.qvel[6:6 + self.n].astype(np.float32)
        ang = self._ang_vel()
        euler = self._quat_to_euler()

        obs = np.concatenate([
            [sin_p, cos_p],
            self.cmd,
            q  * 1.0,
            dq * 0.05,
            ang * 1.0,
            euler * 1.0,
        ])
        return np.clip(obs, -18., 18.).astype(np.float32)

    def _reward(self, action):
        euler = self._quat_to_euler()
        height = float(self.data.qpos[2])
        lin_vel = self.data.qvel[:3].astype(np.float32)

        # velocity tracking
        vel_error = np.sum((lin_vel[:2] - self.cmd[:2]) ** 2)
        r_vel = float(np.exp(-vel_error * 5.))

        # orientation stability
        r_ori = float(np.exp(-np.sum(euler[:2] ** 2) * 10.))

        # base height
        r_height = float(np.exp(-abs(height - self.standing_height) * 100.))

        # action smoothness
        r_smooth = float(-0.002 * np.sum((action - self._last_action) ** 2))

        reward = r_vel + r_ori + r_height + r_smooth
        info = {
            'r_vel': r_vel, 'r_ori': r_ori,
            'r_height': r_height, 'r_smooth': r_smooth,
            'forward_vel': float(lin_vel[0]),
        }
        return reward, info

    def _is_terminated(self):
        height = float(self.data.qpos[2])
        euler  = self._quat_to_euler()
        fallen = height < 0.3 or abs(euler[0]) > 1.0 or abs(euler[1]) > 0.8
        return bool(fallen)
