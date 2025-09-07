# ros2_ws/src/bipedal_robot_description/scripts/robot_controller.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import time

class BipedalRobotController(Node):
    """Basic controller for the bipedal robot"""
    
    def __init__(self):
        super().__init__('bipedal_robot_controller')
        
        # Joint names
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]
        
        # Publishers
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray, 
            '/bipedal_robot/joint_commands', 
            10
        )
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/bipedal_robot/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # State variables
        self.joint_positions = np.zeros(6)
        self.joint_velocities = np.zeros(6)
        self.target_linear_vel = 0.0
        self.target_angular_vel = 0.0
        
        # Control parameters
        self.kp = 100.0  # Proportional gain
        self.kd = 10.0   # Derivative gain
        self.max_torque = 100.0
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100 Hz
        
        self.get_logger().info('Bipedal Robot Controller initialized')
    
    def joint_state_callback(self, msg):
        """Update joint states"""
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.joint_positions[i] = msg.position[idx]
                if len(msg.velocity) > idx:
                    self.joint_velocities[i] = msg.velocity[idx]
    
    def cmd_vel_callback(self, msg):
        """Update velocity commands"""
        self.target_linear_vel = msg.linear.x
        self.target_angular_vel = msg.angular.z
    
    def control_loop(self):
        """Main control loop"""
        # Generate target joint positions based on velocity commands
        target_positions = self.generate_walking_pattern()
        
        # Calculate torques using PD controller
        torques = self.pd_controller(target_positions)
        
        # Publish joint commands
        self.publish_joint_commands(torques)
    
    def generate_walking_pattern(self):
        """Generate basic walking pattern"""
        t = time.time()
        frequency = 1.0  # Walking frequency in Hz
        
        # Simple sinusoidal walking pattern
        phase = 2 * np.pi * frequency * t
        
        # Hip joints (forward/backward swing)
        left_hip = 0.3 * np.sin(phase) * self.target_linear_vel
        right_hip = -0.3 * np.sin(phase) * self.target_linear_vel
        
        # Knee joints (bend during swing phase)
        left_knee = -0.5 * max(0, np.sin(phase)) * abs(self.target_linear_vel)
        right_knee = -0.5 * max(0, -np.sin(phase)) * abs(self.target_linear_vel)
        
        # Ankle joints (slight adjustment for stability)
        left_ankle = 0.1 * np.sin(phase * 2) * self.target_linear_vel
        right_ankle = -0.1 * np.sin(phase * 2) * self.target_linear_vel
        
        return np.array([left_hip, left_knee, left_ankle, 
                        right_hip, right_knee, right_ankle])
    
    def pd_controller(self, target_positions):
        """PD controller for joint torques"""
        position_error = target_positions - self.joint_positions
        velocity_error = -self.joint_velocities  # Assume target velocity is 0
        
        torques = self.kp * position_error + self.kd * velocity_error
        
        # Clip torques to maximum values
        torques = np.clip(torques, -self.max_torque, self.max_torque)
        
        return torques
    
    def publish_joint_commands(self, torques):
        """Publish joint torque commands"""
        msg = Float64MultiArray()
        msg.data = torques.tolist()
        self.joint_cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    controller = BipedalRobotController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

# ros2_ws/src/bipedal_robot_description/launch/complete_simulation.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_bipedal_description = get_package_share_directory('bipedal_robot_description')

    # Launch arguments
    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value=os.path.join(pkg_bipedal_description, 'worlds', 'empty_world.world'),
        description='World file to load in Gazebo'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    enable_gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Enable Gazebo GUI'
    )

    # Gazebo server
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={
            'world': LaunchConfiguration('world_file'),
            'verbose': 'false',
            'pause': 'false'
        }.items()
    )

    # Gazebo client (GUI)
    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        ),
        condition=LaunchConfiguration('gui')
    )

    # Robot description
    urdf_file = os.path.join(pkg_bipedal_description, 'urdf', 'bipedal_robot.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }],
        output='screen'
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # Spawn robot in Gazebo
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_bipedal_robot',
        arguments=[
            '-entity', 'bipedal_robot',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.0'
        ],
        output='screen'
    )

    # Robot controller
    robot_controller = Node(
        package='bipedal_robot_description',
        executable='robot_controller.py',
        name='bipedal_robot_controller',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    return LaunchDescription([
        world_file_arg,
        use_sim_time_arg,
        enable_gui_arg,
        gazebo_server,
        gazebo_client,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        robot_controller
    ])

# src/rl_bipedal_walking/visualization/plot_utils.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Any
import pandas as pd

class TrainingVisualizer:
    """Utilities for visualizing training progress and results"""
    
    def __init__(self, style='seaborn-v0_8'):
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 8)
    
    def plot_training_curves(self, metrics: Dict[str, List], save_path: str = None):
        """Plot comprehensive training curves"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Episode rewards
        if 'episode_rewards' in metrics:
            rewards = metrics['episode_rewards']
            axes[0, 0].plot(rewards, color=self.colors[0], alpha=0.7)
            
            # Moving average
            if len(rewards) > 50:
                window = min(50, len(rewards) // 10)
                moving_avg = pd.Series(rewards).rolling(window=window).mean()
                axes[0, 0].plot(moving_avg, color=self.colors[1], linewidth=2, label=f'MA({window})')
                axes[0, 0].legend()
            
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        if 'episode_lengths' in metrics:
            lengths = metrics['episode_lengths']
            axes[0, 1].plot(lengths, color=self.colors[2], alpha=0.7)
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Policy loss
        if 'policy_losses' in metrics:
            losses = metrics['policy_losses']
            axes[0, 2].plot(losses, color=self.colors[3], alpha=0.7)
            axes[0, 2].set_title('Policy Loss')
            axes[0, 2].set_xlabel('Update')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Value loss
        if 'value_losses' in metrics:
            v_losses = metrics['value_losses']
            axes[1, 0].plot(v_losses, color=self.colors[4], alpha=0.7)
            axes[1, 0].set_title('Value Loss')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Success rate (if applicable)
        if 'success_rates' in metrics:
            success = metrics['success_rates']
            axes[1, 1].plot(success, color=self.colors[5], alpha=0.7)
            axes[1, 1].set_title('Success Rate')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
        
        # Learning progress (reward distribution over time)
        if 'episode_rewards' in metrics:
            rewards = metrics['episode_rewards']
            if len(rewards) > 100:
                # Split into chunks and plot distribution
                chunk_size = len(rewards) // 5
                chunks = [rewards[i:i+chunk_size] for i in range(0, len(rewards), chunk_size)]
                
                for i, chunk in enumerate(chunks):
                    if len(chunk) > 0:
                        axes[1, 2].hist(chunk, bins=20, alpha=0.6, 
                                       label=f'Episodes {i*chunk_size}-{(i+1)*chunk_size}',
                                       color=self.colors[i])
                
                axes[1, 2].set_title('Reward Distribution Evolution')
                axes[1, 2].set_xlabel('Reward')
                axes[1, 2].set_ylabel('Frequency')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        return fig
    
    def plot_episode_analysis(self, trajectory: List[Dict], save_path: str = None):
        """Detailed analysis of a single episode"""
        steps = [d['step'] for d in trajectory]
        positions = np.array([d['position'] for d in trajectory])
        orientations = np.array([d['orientation'] for d in trajectory])
        actions = np.array([d['action'] for d in trajectory])
        rewards = [d['reward'] for d in trajectory]
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 3D trajectory
        ax = fig.add_subplot(3, 3, 1, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                color=self.colors[0], linewidth=2)
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  color='green', s=100, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  color='red', s=100, label='End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory')
        ax.legend()
        
        # Remove the 2D subplot that was there before
        axes[0, 0].remove()
        
        # XY trajectory  
        axes[0, 1].plot(positions[:, 0], positions[:, 1], color=self.colors[0], linewidth=2)
        axes[0, 1].scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start')
        axes[0, 1].scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End')
        axes[0, 1].set_xlabel('X Position (m)')
        axes[0, 1].set_ylabel('Y Position (m)')
        axes[0, 1].set_title('XY Trajectory')
        axes[0, 1].axis('equal')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Height over time
        axes[0, 2].plot(steps, positions[:, 2], color=self.colors[1], linewidth=2)
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Height (m)')
        axes[0, 2].set_title('Height over Time')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Orientation angles
        for i, (angle, name) in enumerate(zip(orientations.T, ['Roll', 'Pitch', 'Yaw'])):
            axes[1, 0].plot(steps, angle, color=self.colors[i+2], label=name, linewidth=2)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Angle (rad)')
        axes[1, 0].set_title('Orientation over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Action history (joint commands)
        joint_names = ['L_Hip', 'L_Knee', 'L_Ankle', 'R_Hip', 'R_Knee', 'R_Ankle']
        for i in range(min(6, actions.shape[1])):
            axes[1, 1].plot(steps, actions[:, i], color=self.colors[i], 
                           label=joint_names[i], linewidth=1, alpha=0.8)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Action Value')
        axes[1, 1].set_title('Action History')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Rewards over time
        axes[1, 2].plot(steps, rewards, color=self.colors[0], linewidth=2)
        axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].set_title('Rewards over Time')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Cumulative reward
        cumulative_rewards = np.cumsum(rewards)
        axes[2, 0].plot(steps, cumulative_rewards, color=self.colors[1], linewidth=2)
        axes[2, 0].set_xlabel('Step')
        axes[2, 0].set_ylabel('Cumulative Reward')
        axes[2, 0].set_title('Cumulative Reward')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Velocity analysis
        if len(positions) > 1:
            velocities = np.diff(positions[:, 0])  # Forward velocity approximation
            axes[2, 1].plot(steps[1:], velocities, color=self.colors[2], linewidth=2)
            axes[2, 1].set_xlabel('Step')
            axes[2, 1].set_ylabel('Forward Velocity (m/step)')
            axes[2, 1].set_title('Forward Velocity')
            axes[2, 1].grid(True, alpha=0.3)
        
        # Stability analysis (orientation magnitude)
        stability = np.linalg.norm(orientations[:, :2], axis=1)  # Roll + Pitch magnitude
        axes[2, 2].plot(steps, stability, color=self.colors[3], linewidth=2)
        axes[2, 2].set_xlabel('Step')
        axes[2, 2].set_ylabel('Orientation Magnitude (rad)')
        axes[2, 2].set_title('Stability (Roll + Pitch)')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Episode analysis saved to {save_path}")
        
        return fig
    
    def plot_hyperparameter_comparison(self, results: Dict[str, Dict], save_path: str = None):
        """Compare results across different hyperparameters"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        labels = list(results.keys())
        final_rewards = [np.mean(results[label]['rewards'][-100:]) if len(results[label]['rewards']) >= 100 
                        else np.mean(results[label]['rewards']) for label in labels]
        max_rewards = [np.max(results[label]['rewards']) for label in labels]
        avg_lengths = [np.mean(results[label]['lengths']) for label in labels]
        convergence_episodes = [len(results[label]['rewards']) for label in labels]
        
        # Bar plots
        x_pos = np.arange(len(labels))
        
        axes[0, 0].bar(x_pos, final_rewards, color=self.colors[:len(labels)])
        axes[0, 0].set_xlabel('Configuration')
        axes[0, 0].set_ylabel('Final Average Reward')
        axes[0, 0].set_title('Final Performance Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(labels, rotation=45)
        
        axes[0, 1].bar(x_pos, max_rewards, color=self.colors[:len(labels)])
        axes[0, 1].set_xlabel('Configuration')
        axes[0, 1].set_ylabel('Maximum Reward')
        axes[0, 1].set_title('Best Performance Comparison')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(labels, rotation=45)
        
        axes[1, 0].bar(x_pos, avg_lengths, color=self.colors[:len(labels)])
        axes[1, 0].set_xlabel('Configuration')
        axes[1, 0].set_ylabel('Average Episode Length')
        axes[1, 0].set_title('Episode Length Comparison')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(labels, rotation=45)
        
        # Learning curves comparison
        for i, label in enumerate(labels):
            rewards = results[label]['rewards']
            if len(rewards) > 10:
                window = min(50, len(rewards) // 10)
                smoothed = pd.Series(rewards).rolling(window=window).mean()
                axes[1, 1].plot(smoothed, color=self.colors[i], label=label, linewidth=2)
        
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Smoothed Reward')
        axes[1, 1].set_title('Learning Curves Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Hyperparameter comparison saved to {save_path}")
        
        return fig
        