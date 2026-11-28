import math

import numpy as np

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.mujoco_rl.mujoco_humanoid_env import MujocoHumanoidEnv
from humanoid.scripts.eval_commands import evaluate

XML_PATH = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-train.xml'


def test_evaluate_reports_displacement_and_falls_per_command():
    env = MujocoHumanoidEnv(XML_PATH, episode_length_s=0.2)
    rows = evaluate(env, lambda obs: np.zeros(12), [0.0, 0.2], episodes=2)
    env.close()
    assert [r['cmd'] for r in rows] == [0.0, 0.2]
    assert math.isnan(rows[0]['disp_ratio'])
    standing = rows[1]
    assert standing['fall_rate'] == 0.0
    assert standing['survived_s'] == 0.2
    assert abs(standing['displacement']) < 0.05
    assert standing['disp_ratio'] == standing['horizon_ratio']
