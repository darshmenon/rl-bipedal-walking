import numpy as np
import pytest

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.mujoco_rl.mujoco_humanoid_env import LIMIT_PROFILES, MujocoHumanoidEnv

sim2sim = pytest.importorskip('humanoid.scripts.sim2sim')

XML_PATH = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-train.xml'


@pytest.mark.parametrize('limits', list(LIMIT_PROFILES))
def test_sim2sim_limits_match_env(limits):
    env = MujocoHumanoidEnv(XML_PATH, limits=limits)
    cfg = sim2sim.make_cfg(limits=limits).robot_config
    np.testing.assert_array_equal(cfg.tau_limit, env.tau_limit)
    assert cfg.clip_actions == env.clip_actions == env.action_space.high[0]
    env.close()


def test_legacy_profile_keeps_old_values():
    assert LIMIT_PROFILES['legacy']['clip_actions'] == 18.0
    assert np.all(LIMIT_PROFILES['legacy']['tau_limit'] == 200.0)
