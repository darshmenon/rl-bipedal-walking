# src/rl_bipedal_walking/training/train_walker.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from datetime import datetime
import rclpy

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.bipedal_env import BipedalWalkingEnv
from agents.ppo_agent import PPOAgent

class WalkingTrainer:
    """Trainer class for bipedal walking RL"""
    
    def __init__(self, config):
        self.config = config
        
        # Create environment
        print("Initializing environment...")
        self.env = BipedalWalkingEnv()
        
        # Create agent
        print("Initializing agent...")
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=config.learning_rate,
            gamma=config.gamma,
            eps_clip=config.eps_clip,
            k_epochs=config.k_epochs,
            entropy_coef=config.entropy_coef
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        
        # Create save directories
        self.model_dir = os.path.join(config.save_dir, "models")
        self.plot_dir = os.path.join(config.save_dir, "plots")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.max_episodes} episodes...")
        
        best_reward = -np.inf
        
        for episode in range(self.config.max_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            print(f"\nEpisode {episode + 1}/{self.config.max_episodes}")
            
            while True:
                # Select action
                action, log_prob, value = self.agent.select_action(state)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.agent.store_transition(
                    state, action, log_prob, reward, 
                    value.item() if value is not None else 0.0, done
                )
                
                # Update metrics
                episode_reward += reward
                episode_length += 1
                
                # Move to next state
                state = next_state
                
                # Print progress every 100 steps
                if episode_length % 100 == 0:
                    pos = info['robot_position']
                    vel = info['forward_velocity']
                    print(f"  Step {episode_length}: Pos=({pos[0]:.2f}, {pos[2]:.2f}), "
                          f"Vel={vel:.2f}, Reward={reward:.2f}")
                
                if done:
                    break
            
            # Update policy
            if len(self.agent.states) > 0:
                update_info = self.agent.update(next_state if not done else None)
                if update_info:
                    self.policy_losses.append(update_info['policy_loss'])
                    self.value_losses.append(update_info['value_loss'])
            
            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Print episode summary
            print(f"Episode {episode + 1} completed:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length}")
            print(f"  Final position: {info['robot_position']}")
            print(f"  Average reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                model_path = os.path.join(self.model_dir, "best_model.pth")
                self.agent.save_model(model_path)
                print(f"  New best model saved! Reward: {best_reward:.2f}")
            
            # Save model periodically
            if (episode + 1) % self.config.save_interval == 0:
                model_path = os.path.join(self.model_dir, f"model_episode_{episode + 1}.pth")
                self.agent.save_model(model_path)
                print(f"  Model saved at episode {episode + 1}")
            
            # Plot progress periodically
            if (episode + 1) % self.config.plot_interval == 0:
                self.plot_progress()
                print(f"  Progress plots updated")
        
        print("\nTraining completed!")
        self.plot_progress()
        
    def plot_progress(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Moving average of rewards
        if len(self.episode_rewards) > 10:
            window_size = min(100, len(self.episode_rewards))
            moving_avg = np.convolve(self.episode_rewards, 
                                   np.ones(window_size)/window_size, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Moving Average Rewards (window={window_size})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].grid(True)
        
        # Episode lengths
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # Policy loss
        if self.policy_losses:
            axes[1, 1].plot(self.policy_losses)
            axes[1, 1].set_title('Policy Loss')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plot_dir, f'training_progress_{len(self.episode_rewards)}.png')
        plt.savefig(plot_path)
        plt.close()

class TrainingConfig:
    """Training configuration"""
    def __init__(self):
        # Environment parameters
        self.max_episodes = 1000
        
        # Agent parameters
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.k_epochs = 10
        self.entropy_coef = 0.01
        
        # Training parameters
        self.save_interval = 50
        self.plot_interval = 25
        self.save_dir = "training_results"

def main():
    parser = argparse.ArgumentParser(description='Train bipedal walking robot')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='training_results', help='Directory to save results')
    parser.add_argument('--save-interval', type=int, default=50, help='Model save interval')
    parser.add_argument('--plot-interval', type=int, default=25, help='Plot update interval')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    config.max_episodes = args.episodes
    config.learning_rate = args.lr
    config.save_dir = args.save_dir
    config.save_interval = args.save_interval
    config.plot_interval = args.plot_interval
    
    # Add timestamp to save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.save_dir = f"{config.save_dir}_{timestamp}"
    
    try:
        # Initialize and start training
        trainer = WalkingTrainer(config)
        trainer.train()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        # Clean up
        try:
            trainer.env.close()
        except:
            pass
        print("Training session ended")

if __name__ == "__main__":
    main()