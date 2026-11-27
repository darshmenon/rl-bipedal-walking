import numpy as np
import pytest

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.mujoco_rl.mujoco_humanoid_env import CMD_VX_RANGE, MujocoHumanoidEnv

XML_PATH = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-train.xml'


def make(**kwargs):
    return MujocoHumanoidEnv(XML_PATH, **kwargs)


def test_command_is_fixed_by_default():
    env = make(cmd_vx=0.3)
    for seed in range(3):
        env.reset(seed=seed)
        assert env.cmd_vx == 0.3


def test_randomized_command_varies_within_range_and_is_seeded():
    env = make(randomize_cmd=True)
    env.reset(seed=0)
    first = env.cmd_vx
    values = [first]
    for _ in range(20):
        env.reset()
        values.append(env.cmd_vx)
    assert len(set(values)) > 1
    assert all(CMD_VX_RANGE[0] <= v <= CMD_VX_RANGE[1] for v in values)
    env.reset(seed=0)
    assert env.cmd_vx == first


def test_observation_carries_sampled_command():
    env = make(randomize_cmd=True)
    obs, _ = env.reset(seed=1)
    newest = obs.reshape(15, -1)[-1]
    assert newest[2] == pytest.approx(env.cmd_vx * 2.0)


@pytest.mark.parametrize('limits', ['current', 'legacy'])
def test_applied_torque_and_action_respect_limits(limits):
    env = make(limits=limits)
    env.reset(seed=0)
    env.step(np.full(12, 100.0))
    assert np.all(np.abs(env.data.ctrl) <= env.tau_limit + 1e-9)
    assert np.all(np.abs(env.prev_action) <= env.clip_actions)
