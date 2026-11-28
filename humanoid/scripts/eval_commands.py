"""
Evaluate a policy per commanded forward speed: net displacement relative to the
command, and fall rate. Use it to check that a policy really walks (rocking in
place gives a high reward-term average but little net displacement).

Usage:
    python humanoid/scripts/eval_commands.py --model logs/<run>/final_model.zip
    python humanoid/scripts/eval_commands.py --model <...>/policy_1.pt --limits legacy
"""
import argparse
import os
import sys

sys.modules['tensorflow'] = None  # see train_mujoco.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.mujoco_rl.mujoco_humanoid_env import MujocoHumanoidEnv, LIMIT_PROFILES

DEFAULT_COMMANDS = [-0.3, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def load_policy(path):
    """Return obs -> action for an SB3 .zip checkpoint or an exported TorchScript .pt."""
    if path.endswith('.zip'):
        from stable_baselines3 import PPO
        model = PPO.load(path, device='cpu')
        return lambda obs: model.predict(obs, deterministic=True)[0]
    import torch
    module = torch.jit.load(path, map_location='cpu').eval()

    def act(obs):
        with torch.no_grad():
            return module(torch.from_numpy(obs[None]))[0].numpy()
    return act


def evaluate(env, policy, commands, episodes, seed=0):
    """
    For each command return a dict with:
      disp_ratio     net x displacement / (cmd * survived time); 1 = on command
      horizon_ratio  net x displacement / (cmd * full episode); falls count against it
      fall_rate      fraction of episodes ending in a fall
    Ratios are NaN for a zero command.
    """
    dt = env.sim_dt * env.decimation
    rows = []
    for cmd in commands:
        disp, times, falls = [], [], 0
        for ep in range(episodes):
            env.cmd_vx = cmd
            obs, _ = env.reset(seed=seed + ep)
            x0 = env.data.qpos[0]
            steps, terminated, truncated = 0, False, False
            while not (terminated or truncated):
                obs, _, terminated, truncated, _ = env.step(policy(obs))
                steps += 1
            disp.append(env.data.qpos[0] - x0)
            times.append(steps * dt)
            falls += int(terminated)
        horizon = env.max_steps * dt
        rows.append(dict(
            cmd=cmd,
            displacement=float(np.mean(disp)),
            survived_s=float(np.mean(times)),
            disp_ratio=float(np.sum(disp) / (cmd * np.sum(times))) if cmd else float('nan'),
            horizon_ratio=float(np.mean(disp) / (cmd * horizon)) if cmd else float('nan'),
            fall_rate=falls / episodes,
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='SB3 .zip or TorchScript .pt')
    parser.add_argument('--commands', type=float, nargs='+', default=DEFAULT_COMMANDS)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--episode_length_s', type=float, default=24.0)
    parser.add_argument('--limits', choices=list(LIMIT_PROFILES), default='current')
    parser.add_argument('--terrain', action='store_true')
    args = parser.parse_args()

    xml_name = 'XBot-L-terrain.xml' if args.terrain else 'XBot-L-train.xml'
    env = MujocoHumanoidEnv(f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/{xml_name}',
                            episode_length_s=args.episode_length_s, limits=args.limits)
    rows = evaluate(env, load_policy(args.model), args.commands, args.episodes)
    env.close()

    print(f'{"cmd":>6} {"disp_m":>8} {"alive_s":>8} {"disp/cmd":>9} {"horizon":>8} {"falls":>6}')
    for r in rows:
        print(f'{r["cmd"]:>6.2f} {r["displacement"]:>8.2f} {r["survived_s"]:>8.1f} '
              f'{r["disp_ratio"]:>9.2f} {r["horizon_ratio"]:>8.2f} {r["fall_rate"]:>6.0%}')


if __name__ == '__main__':
    main()
