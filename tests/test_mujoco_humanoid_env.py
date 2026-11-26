import os

import numpy as np
import pytest

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.mujoco_rl.mujoco_humanoid_env import MujocoHumanoidEnv, NUM_OBS

XML_PATH = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-train.xml'


@pytest.fixture
def env():
    e = MujocoHumanoidEnv(XML_PATH)
    yield e
    e.close()


def test_reset_returns_stacked_obs(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (NUM_OBS,)
    assert obs.dtype == np.float32
    assert info == {}


def test_reset_is_deterministic_with_seed(env):
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_allclose(obs1, obs2)


def test_step_returns_expected_shapes(env):
    env.reset(seed=0)
    action = np.zeros(12, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (NUM_OBS,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert 'base_height' in info
    assert 'forward_vel' in info
    assert 'lateral_vel' in info
    assert 'roll' in info
    assert 'pitch' in info
    assert 'yaw_rate' in info


def test_alive_bonus_only_reward_stays_positive_near_stand(env):
    env.reset(seed=0)
    action = np.zeros(12, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert not terminated
    assert reward > 0


def test_episode_terminates_when_fallen(env):
    env.reset(seed=0)
    env.data.qpos[2] = -0.5  # offset from base_link rest height (~0.88 m) -> world z ~0.38, below fall threshold
    action = np.zeros(12, dtype=np.float32)
    _, reward, terminated, _, _ = env.step(action)
    assert terminated
    assert reward < 0


def test_episode_truncates_at_max_steps(env):
    env.reset(seed=0)
    action = np.zeros(12, dtype=np.float32)
    for _ in range(env.max_steps - 1):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated:
            pytest.skip('robot fell before reaching max_steps under a zero action')
    _, _, terminated, truncated, _ = env.step(action)
    assert truncated
