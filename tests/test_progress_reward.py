import numpy as np
import pytest

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.mujoco_rl.mujoco_humanoid_env import MujocoHumanoidEnv

XML_PATH = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-train.xml'
DT = 0.01


@pytest.fixture(scope='module')
def env():
    e = MujocoHumanoidEnv(XML_PATH)
    yield e
    e.close()


def terms(env, cmd, velocities):
    env.reset(seed=0)
    env.cmd_vx = cmd
    return np.array([env._progress_reward(v, DT) for v in velocities])


@pytest.mark.parametrize('cmd', [-0.3, 0.1, 0.4])
def test_tracking_the_command_earns_full_term_and_standing_earns_none(env, cmd):
    assert terms(env, cmd, np.full(1000, cmd))[-1] == pytest.approx(1.0, abs=1e-3)
    assert np.all(terms(env, cmd, np.zeros(100)) == 0.0)


def test_overspeed_is_not_paid_extra(env):
    assert terms(env, 0.2, np.full(1000, 1.0)).max() == pytest.approx(1.0)
    assert terms(env, 0.06, np.full(1000, 1.0)).max() == pytest.approx(0.6)


def test_backward_motion_is_floored(env):
    assert terms(env, 0.1, np.full(1000, -2.0)).min() == pytest.approx(-1.0)


def test_off_inside_deadband(env):
    assert np.all(terms(env, 0.03, np.full(100, 0.5)) == 0.0)


def test_fast_rocking_in_place_earns_nothing(env):
    t = np.arange(6000) * DT
    v = np.where((t % 0.64) < 0.576, 0.2, -1.8)
    assert abs(terms(env, 0.2, v)[1000:].mean()) < 0.05


def test_filter_resets_between_episodes(env):
    terms(env, 0.4, np.full(200, 0.4))
    env.reset(seed=0)
    assert env.vx_filtered == 0.0
