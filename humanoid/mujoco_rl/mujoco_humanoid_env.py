"""
Gymnasium-native MuJoCo training environment for the XBot-L humanoid.

Isaac Gym (used by humanoid/scripts/train.py) is a discontinued NVIDIA package
that isn't installable on this machine. This env runs the same XBot-L MJCF
model directly through the `mujoco` Python bindings so training can proceed
without Isaac Gym, using stable-baselines3 instead of the repo's custom
Isaac-Gym-coupled PPO runner.

The observation/action layout intentionally mirrors humanoid/scripts/sim2sim.py
so a policy trained here can be loaded directly by that script for sim-to-sim
validation.
"""
import math
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

JOINT_NAMES = [
    'left_leg_roll_joint', 'left_leg_yaw_joint', 'left_leg_pitch_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_leg_roll_joint', 'right_leg_yaw_joint', 'right_leg_pitch_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
]

KP = np.array([200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], dtype=np.float64)
KD = np.array([10] * 12, dtype=np.float64)
# Order = JOINT_NAMES (hip roll/yaw/pitch, knee, ankle pitch/roll). XBot-L.urdf effort
# limits scaled by 0.85 (safety.torque_limit), matching the Isaac Gym config.
TAU_LIMIT = 0.85 * np.array([100, 100, 250, 250, 100, 100] * 2, dtype=np.float64)
ACTION_SCALE = 0.25
CLIP_ACTIONS = 5.0  # was 18 (PD target up to 4.5 rad); 5*ACTION_SCALE = 1.25 rad
# Action/torque limits shared by training and sim2sim. 'legacy' is for checkpoints
# trained before the per-joint limits (e.g. to evaluate or resume them).
LIMIT_PROFILES = {
    'current': dict(tau_limit=TAU_LIMIT, clip_actions=CLIP_ACTIONS),
    'legacy': dict(tau_limit=np.full(12, 200.0), clip_actions=18.0),
}
CMD_VX_RANGE = (-0.3, 0.6)  # matches humanoid_config.commands.ranges.lin_vel_x
CLIP_OBS = 18.0
CYCLE_TIME = 0.64
BASE_HEIGHT_TARGET = 0.89
FALL_HEIGHT = 0.5
FALL_ROLL_PITCH = 0.65
# Lateral CoM sway: shift weight toward the stance leg each half-cycle instead
# of just penalizing lateral drift, mirroring the ZMP sway target in bheema's
# com_traj.py (low-pass filtered so the target doesn't teleport at phase switches).
STANCE_WIDTH = 0.2
SWAY_FRACTION = 0.6
SWAY_FILTER_ALPHA = 0.85
NUM_SINGLE_OBS = 47
FRAME_STACK = 15
NUM_OBS = FRAME_STACK * NUM_SINGLE_OBS

OBS_SCALES = dict(lin_vel=2.0, ang_vel=1.0, dof_pos=1.0, dof_vel=0.05)


def quat_to_euler(quat_xyzw):
    x, y, z, w = quat_xyzw
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return np.array([roll, pitch, yaw])


class MujocoHumanoidEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, xml_path, render_mode=None, sim_dt=0.001, decimation=10,
                 episode_length_s=24.0, cmd_vx=0.4, cmd_vy=0.0, cmd_dyaw=0.0, randomize_cmd=False, limits='current'):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = sim_dt
        self.data = mujoco.MjData(self.model)
        self.decimation = decimation
        self.sim_dt = sim_dt
        self.max_steps = int(episode_length_s / (sim_dt * decimation))
        self.render_mode = render_mode
        self.viewer = None

        self.cmd_vx, self.cmd_vy, self.cmd_dyaw = cmd_vx, cmd_vy, cmd_dyaw
        self.randomize_cmd = randomize_cmd
        self.tau_limit = LIMIT_PROFILES[limits]['tau_limit']
        self.clip_actions = LIMIT_PROFILES[limits]['clip_actions']

        self.joint_qpos_adr = np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in JOINT_NAMES])
        self.joint_qvel_adr = np.array(
            [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in JOINT_NAMES])

        self.action_space = spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-CLIP_OBS, high=CLIP_OBS, shape=(NUM_OBS,), dtype=np.float32)

        self.base_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        self._init_qpos = None
        self._init_qvel = None
        self.hist_obs = None
        self.prev_action = None
        self.count = 0
        self.lateral_sway_target = 0.0

    def _get_state(self):
        q = self.data.qpos[self.joint_qpos_adr].copy()
        dq = self.data.qvel[self.joint_qvel_adr].copy()
        quat_wxyz = self.data.sensor('orientation').data
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        omega = self.data.sensor('angular-velocity').data.copy()
        return q, dq, quat_xyzw, omega

    def _build_obs(self):
        q, dq, quat_xyzw, omega = self._get_state()
        eu_ang = quat_to_euler(quat_xyzw)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        obs = np.zeros(NUM_SINGLE_OBS, dtype=np.float32)
        t = self.count * self.sim_dt * self.decimation
        obs[0] = math.sin(2 * math.pi * t / CYCLE_TIME)
        obs[1] = math.cos(2 * math.pi * t / CYCLE_TIME)
        obs[2] = self.cmd_vx * OBS_SCALES['lin_vel']
        obs[3] = self.cmd_vy * OBS_SCALES['lin_vel']
        obs[4] = self.cmd_dyaw * OBS_SCALES['ang_vel']
        obs[5:17] = q * OBS_SCALES['dof_pos']
        obs[17:29] = dq * OBS_SCALES['dof_vel']
        obs[29:41] = self.prev_action
        obs[41:44] = omega
        obs[44:47] = eu_ang
        obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)

        self.hist_obs.append(obs)
        self.hist_obs.pop(0)
        return np.concatenate(self.hist_obs).astype(np.float32)

    def reset(self, *, seed=None, restart=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # base_link's <body pos="0 0 0.88"> is baked into forward kinematics
        # separately from qpos -- the free joint's qpos[2] is a TRANSLATION ON
        # TOP OF that pos, not an absolute world height. Read the compiled
        # rest height first so the spawn-height override below lands at the
        # intended *world* height regardless of what the loaded MJCF bakes
        # into base_link's pos (rather than hardcoding "0.88" and silently
        # breaking again if a different robot XML is ever loaded here).
        mujoco.mj_forward(self.model, self.data)
        rest_height = self.data.xpos[self.base_bid][2]
        self.data.qpos[2] = 0.95 - rest_height
        self.data.qpos[3] = 1.0  # quat w
        for adr in self.joint_qpos_adr:
            self.data.qpos[adr] = 0.0 + self.np_random.uniform(-0.02, 0.02)
        mujoco.mj_forward(self.model, self.data)

        if self.randomize_cmd:
            self.cmd_vx = float(self.np_random.uniform(*CMD_VX_RANGE))
        self.prev_action = np.zeros(12, dtype=np.float32)
        self.hist_obs = [np.zeros(NUM_SINGLE_OBS, dtype=np.float32) for _ in range(FRAME_STACK)]
        self.count = 0
        self.lateral_sway_target = 0.0
        obs = self._build_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(action, -self.clip_actions, self.clip_actions).astype(np.float64)
        target_q = action * ACTION_SCALE

        for _ in range(self.decimation):
            q = self.data.qpos[self.joint_qpos_adr]
            dq = self.data.qvel[self.joint_qvel_adr]
            tau = (target_q - q) * KP + (0.0 - dq) * KD
            tau_demand = tau
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        self.count += 1
        q, dq, quat_xyzw, omega = self._get_state()
        eu_ang = quat_to_euler(quat_xyzw)
        base_y = self.data.qpos[1]
        base_z = self.data.xpos[self.base_bid][2]  # world height; qpos[2] is offset by base_link's XML pos, see reset()
        base_vel = self.data.qvel[:3]

        # Phase-locked lateral sway target: shift toward the stance leg each
        # half-cycle (left stance for the first half, right for the second),
        # low-pass filtered so the target moves rather than teleporting at
        # the phase switch.
        dt_ctrl = self.sim_dt * self.decimation
        phase = (self.count * dt_ctrl % CYCLE_TIME) / CYCLE_TIME
        zmp_target_y = (STANCE_WIDTH / 2.0) * SWAY_FRACTION if phase < 0.5 else -(STANCE_WIDTH / 2.0) * SWAY_FRACTION
        prev_sway_target = self.lateral_sway_target
        self.lateral_sway_target = SWAY_FILTER_ALPHA * prev_sway_target + (1.0 - SWAY_FILTER_ALPHA) * zmp_target_y
        sway_vel_target = (self.lateral_sway_target - prev_sway_target) / dt_ctrl

        forward_vel_err = base_vel[0] - self.cmd_vx
        lateral_vel_err = base_vel[1] - (self.cmd_vy + sway_vel_target)
        lateral_pos_err = base_y - self.lateral_sway_target
        yaw_rate_err = omega[2] - self.cmd_dyaw
        tilt_err = eu_ang[0] ** 2 + eu_ang[1] ** 2
        wobble_err = omega[0] ** 2 + omega[1] ** 2
        reward = 0.0
        reward += 1.2 * math.exp(-5.0 * forward_vel_err ** 2)
        reward += 0.5 * math.exp(-8.0 * lateral_vel_err ** 2)
        reward += 0.6 * math.exp(-30.0 * lateral_pos_err ** 2)
        reward += 0.4 * math.exp(-6.0 * yaw_rate_err ** 2)
        reward += 1.4 * math.exp(-12.0 * tilt_err)
        reward += 0.4 * math.exp(-2.0 * wobble_err)
        reward += 0.2 * math.exp(-20.0 * (base_z - BASE_HEIGHT_TARGET) ** 2)
        reward -= 1e-5 * float(np.sum(np.square(tau_demand)))  # unclipped PD demand (last substep)
        reward -= 0.002 * float(np.sum(np.square(action - self.prev_action)))
        reward += 0.5  # alive bonus

        fell = base_z < FALL_HEIGHT or abs(eu_ang[0]) > FALL_ROLL_PITCH or abs(eu_ang[1]) > FALL_ROLL_PITCH
        if fell:
            reward -= 10.0

        self.prev_action = action.astype(np.float32)
        obs = self._build_obs()

        terminated = bool(fell)
        truncated = bool(self.count >= self.max_steps)
        return obs, reward, terminated, truncated, {
            "base_height": base_z,
            "forward_vel": base_vel[0],
            "lateral_vel": base_vel[1],
            "lateral_sway_target": self.lateral_sway_target,
            "roll": eu_ang[0],
            "pitch": eu_ang[1],
            "yaw_rate": omega[2],
        }

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            import mujoco_viewer
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
