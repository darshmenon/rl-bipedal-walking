"""
Train the XBot-L humanoid to walk directly in MuJoCo with stable-baselines3 PPO.

This is the Isaac-Gym-free training path: humanoid/scripts/train.py requires
the discontinued `isaacgym` package, which isn't installable here. This script
uses the same XBot-L MJCF model and PD/observation/action conventions (see
humanoid/envs/custom/mujoco_humanoid_env.py) but drives training through
MuJoCo + SB3 PPO with CPU-parallel envs instead of Isaac Gym's GPU-parallel envs.

Usage:
    python humanoid/scripts/train_mujoco.py --run_name v1 --num_envs 16 --total_timesteps 2000000

The exported policy (logs/<run_name>/exported/policies/policy_1.pt) is
directly loadable by humanoid/scripts/sim2sim.py for validation.
"""
import argparse
import os
import sys

# stable-baselines3's tensorboard logging path pulls in the full `tensorflow`
# package if it's installed. On this machine, having TensorFlow and PyTorch
# loaded in the same process segfaults deep inside libc as soon as a PPO
# optimizer is constructed (reproducible even with a plain CartPole env, no
# MuJoCo involved). Blocking the import makes SB3 fall back to its own
# lightweight event writer instead, which sidesteps the crash entirely.
sys.modules['tensorflow'] = None

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import glob
import signal
import time

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.mujoco_rl.mujoco_humanoid_env import MujocoHumanoidEnv, NUM_OBS, LIMIT_PROFILES


class ProgressCallback(BaseCallback):
    """Unattended-run helper: periodic progress line, best-model save, checkpoint
    pruning (keep newest N), and a clean stop on SIGTERM/SIGINT."""

    def __init__(self, log_dir, keep=3, every_s=180):
        super().__init__()
        self.log_dir, self.keep, self.every_s = log_dir, keep, every_s
        self.last = 0.0
        self.best = -float('inf')
        self.stop = False
        signal.signal(signal.SIGTERM, self._sig)
        signal.signal(signal.SIGINT, self._sig)

    def _sig(self, *_):
        self.stop = True

    def _on_training_end(self):
        self._log(final=True)

    def _on_step(self):
        now = time.time()
        if now - self.last >= self.every_s:
            self.last = now
            self._log()
        return not self.stop

    def _log(self, final=False):
        buf = self.model.ep_info_buffer
        rew = float(np.mean([e['r'] for e in buf])) if buf else float('nan')
        ln = float(np.mean([e['l'] for e in buf])) if buf else float('nan')
        with open(f'{self.log_dir}/progress.log', 'a') as f:
            f.write(f'{time.strftime("%F %T")} steps={self.num_timesteps} ep_rew={rew:.1f} ep_len={ln:.0f}\n')
        if buf and len(buf) >= 20 and rew > self.best:
            self.best = rew
            self.model.save(f'{self.log_dir}/best_model')
        ckpts = sorted(glob.glob(f'{self.log_dir}/checkpoints/*.zip'), key=os.path.getmtime)
        for old in ckpts[:-self.keep]:
            os.remove(old)


def make_env(xml_path, randomize_cmd=False, limits='current'):
    def _init():
        return Monitor(MujocoHumanoidEnv(xml_path, randomize_cmd=randomize_cmd, limits=limits))
    return _init


class PolicyExportWrapper(nn.Module):
    """Wraps the SB3 actor so torch.jit.load(...) in sim2sim.py gets a plain
    [1, NUM_OBS] -> [1, 12] module, matching what that script expects."""

    def __init__(self, policy):
        super().__init__()
        self.mlp_extractor = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, obs):
        latent = self.mlp_extractor(obs)
        return self.action_net(latent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='v1')
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--total_timesteps', type=int, default=2_000_000)
    parser.add_argument('--terrain', action='store_true')
    parser.add_argument('--randomize_cmd', action='store_true', help='resample cmd_vx each episode')
    parser.add_argument('--limits', choices=list(LIMIT_PROFILES), default='current',
                        help="use 'legacy' to resume checkpoints trained with clip 18 / torque 200")
    parser.add_argument('--resume', type=str, default=None, help='path to a .zip checkpoint to resume from')
    args = parser.parse_args()

    xml_name = 'XBot-L-terrain.xml' if args.terrain else 'XBot-L-train.xml'
    xml_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/{xml_name}'

    log_dir = f'{LEGGED_GYM_ROOT_DIR}/logs/XBot_mujoco_{args.run_name}'
    os.makedirs(log_dir, exist_ok=True)

    vec_cls = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv
    env = vec_cls([make_env(xml_path, args.randomize_cmd, args.limits) for _ in range(args.num_envs)])

    policy_kwargs = dict(net_arch=dict(pi=[512, 256, 128], vf=[768, 256, 128]))

    if args.resume:
        model = PPO.load(args.resume, env=env, device='cpu')
    else:
        model = PPO(
            'MlpPolicy', env,
            policy_kwargs=policy_kwargs,
            learning_rate=1e-4,
            gamma=0.994,
            gae_lambda=0.9,
            clip_range=0.2,
            target_kl=0.02,
            ent_coef=0.001,
            n_steps=200,
            batch_size=200 * args.num_envs // 4,
            n_epochs=5,
            tensorboard_log=log_dir,
            verbose=1,
            # A tiny MLP policy gains nothing from GPU, and SB3's default
            # 'auto' device picks CUDA here, which drives torch into a
            # triton-fused-Adam codegen path that hangs/segfaults on this
            # machine's driver+torch combo. CPU sidesteps it entirely.
            device='cpu',
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.num_envs, 1),
        save_path=f'{log_dir}/checkpoints',
        name_prefix='ppo_xbot',
    )

    progress_cb = ProgressCallback(log_dir)
    model.learn(total_timesteps=args.total_timesteps, callback=[checkpoint_cb, progress_cb], progress_bar=False)
    model.save(f'{log_dir}/final_model')

    export_dir = f'{log_dir}/exported/policies'
    os.makedirs(export_dir, exist_ok=True)
    export_module = PolicyExportWrapper(model.policy).to('cpu').eval()
    example_input = torch.zeros(1, NUM_OBS)
    traced = torch.jit.trace(export_module, example_input)
    traced.save(f'{export_dir}/policy_1.pt')
    print(f'Exported policy to {export_dir}/policy_1.pt')
    print(f'Validate with: python humanoid/scripts/sim2sim.py --load_model {export_dir}/policy_1.pt')


if __name__ == '__main__':
    main()
