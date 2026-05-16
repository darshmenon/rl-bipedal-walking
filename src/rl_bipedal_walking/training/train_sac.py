"""
train_sac.py

Trains a SAC agent on the MuJoCo bipedal walking environment.
No Isaac Gym or ROS2 required.

Usage:
  python src/rl_bipedal_walking/training/train_sac.py --robot h1
  python src/rl_bipedal_walking/training/train_sac.py --robot xbot --steps 1000000
  python src/rl_bipedal_walking/training/train_sac.py --robot h1 --use_wandb
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.sac_agent import SACAgent
from environments.mujoco_env import MujocoBipedalEnv


def train(args):
    run_name = f"{args.robot}_sac_{datetime.now().strftime('%b%d_%H-%M-%S')}"
    log_dir  = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    env = MujocoBipedalEnv(robot=args.robot)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        auto_alpha=True,
        batch_size=args.batch_size,
        replay_capacity=1_000_000,
        device=args.device,
    )

    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    if args.use_wandb:
        import wandb
        wandb.init(project='bipedal-sac', name=run_name, sync_tensorboard=True,
                   config=vars(args))

    obs, _ = env.reset()
    ep_reward = 0.
    ep_len    = 0
    ep_num    = 0
    best_ep   = -np.inf

    print(f"Training SAC on {args.robot} — {state_dim}D obs / {action_dim}D act")
    print(f"Logs: {log_dir}")

    for step in range(1, args.steps + 1):
        # warm-up: random actions for the first `start_steps`
        if step < args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        ep_reward += reward
        ep_len    += 1

        if step >= args.start_steps:
            metrics = agent.update()
            if metrics and step % 1000 == 0:
                writer.add_scalar('Loss/critic', metrics['critic_loss'], step)
                writer.add_scalar('Loss/policy', metrics['policy_loss'], step)
                writer.add_scalar('SAC/alpha',   metrics['alpha'],       step)

        if done:
            ep_num += 1
            writer.add_scalar('Train/ep_reward', ep_reward, step)
            writer.add_scalar('Train/ep_length', ep_len,    step)
            writer.add_scalar('Train/forward_vel',
                              info.get('forward_vel', 0.), step)

            if ep_num % 10 == 0:
                print(f"  step={step:>7d}  ep={ep_num:>5d}  "
                      f"reward={ep_reward:>8.2f}  len={ep_len}")

            if ep_reward > best_ep:
                best_ep = ep_reward
                agent.save(os.path.join(log_dir, 'best_model.pt'))

            if ep_num % args.save_interval == 0:
                agent.save(os.path.join(log_dir, f'model_ep{ep_num}.pt'))

            obs, _ = env.reset()
            ep_reward = 0.
            ep_len    = 0

    agent.save(os.path.join(log_dir, 'final_model.pt'))
    env.close()
    writer.close()
    print(f"\nTraining done. Best episode reward: {best_ep:.2f}")
    print(f"Models saved to: {log_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--robot',          type=str,   default='h1',
                   choices=['xbot', 'h1'], help='Robot model to train on.')
    p.add_argument('--steps',          type=int,   default=1_000_000)
    p.add_argument('--start_steps',    type=int,   default=10_000,
                   help='Random-action warm-up steps before learning starts.')
    p.add_argument('--lr',             type=float, default=3e-4)
    p.add_argument('--gamma',          type=float, default=0.99)
    p.add_argument('--tau',            type=float, default=0.005)
    p.add_argument('--batch_size',     type=int,   default=256)
    p.add_argument('--save_interval',  type=int,   default=100,
                   help='Save checkpoint every N episodes.')
    p.add_argument('--log_dir',        type=str,   default='logs/sac')
    p.add_argument('--device',         type=str,   default='cpu')
    p.add_argument('--use_wandb',      action='store_true')
    return train(p.parse_args())


if __name__ == '__main__':
    main()
