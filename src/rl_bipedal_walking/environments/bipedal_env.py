# src/rl_bipedal_walking/environments/bipedal_env.py
import gym
from gym import spaces
import numpy as np
import rospy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
import time
import math

class BipedalWalkingEnv(gym.Env, Node):
    """
    Custom Gym environment for bipedal robot walking using ROS2 and Gazebo
    """
    
    def __init__(self):
        super(BipedalWalkingEnv, self).__init__()
        
        # Initialize ROS2 node
        rclpy.init()
        Node.__init__(self, 'bipedal_rl_env')
        
        # Joint names
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]
        
        # Action space: continuous control for 6 joints (torques)
        self.action_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(6,),
            dtype=np.float32
        )
        
        # Observation space: joint positions, velocities, and robot pose
        # 6 joint positions + 6 joint velocities + 6 pose (x,y,z,roll,pitch,yaw) = 18
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32
        )
        
        # ROS2 publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/bipedal_robot/cmd_vel',
            10
        )
        
        # Gazebo services
        self.get_model_state_client = self.create_client(
            GetModelState,
            '/gazebo/get_model_state'
        )
        
        self.set_model_state_client = self.create_client(
            SetModelState,
            '/gazebo/set_model_state'
        )
        
        self.reset_simulation_client = self.create_client(
            Empty,
            '/gazebo/reset_simulation'
        )
        
        # State variables
        self.joint_positions = np.zeros(6)
        self.joint_velocities = np.zeros(6)
        self.robot_position = np.zeros(3)
        self.robot_orientation = np.zeros(3)
        
        # Episode parameters
        self.max_episode_steps = 1000
        self.current_step = 0
        self.target_velocity = 0.5  # m/s forward
        
        # Previous position for velocity calculation
        self.prev_position = np.zeros(3)
        self.start_time = time.time()
        
    def joint_state_callback(self, msg):
        """Callback to update joint states"""
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.joint_positions[i] = msg.position[idx]
                self.joint_velocities[i] = msg.velocity[idx]
    
    def get_robot_state(self):
        """Get robot's current pose from Gazebo"""
        request = GetModelState.Request()
        request.model_name = 'bipedal_robot'
        
        try:
            future = self.get_model_state_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            
            if response.success:
                pose = response.pose
                self.robot_position = np.array([
                    pose.position.x,
                    pose.position.y,
                    pose.position.z
                ])
                
                # Convert quaternion to euler angles
                import tf_transformations
                quaternion = [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w
                ]
                self.robot_orientation = np.array(
                    tf_transformations.euler_from_quaternion(quaternion)
                )
        except Exception as e:
            self.get_logger().error(f"Failed to get robot state: {e}")
    
    def reset(self):
        """Reset the environment"""
        # Reset Gazebo simulation
        try:
            future = self.reset_simulation_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, future)
        except Exception as e:
            self.get_logger().error(f"Failed to reset simulation: {e}")
        
        # Reset episode variables
        self.current_step = 0
        self.start_time = time.time()
        
        # Wait for simulation to stabilize
        time.sleep(1.0)
        
        # Get initial state
        self.get_robot_state()
        self.prev_position = self.robot_position.copy()
        
        return self.get_observation()
    
    def step(self, action):
        """Execute one step in the environment"""
        # Apply action (torques to joints)
        self.apply_action(action)
        
        # Step simulation (let it run for a short time)
        time.sleep(0.02)  # 50 Hz
        rclpy.spin_once(self, timeout_sec=0.001)
        
        # Get new state
        self.get_robot_state()
        observation = self.get_observation()
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Check if episode is done
        done = self.is_done()
        
        self.current_step += 1
        
        # Info dict
        info = {
            'robot_position': self.robot_position,
            'robot_orientation': self.robot_orientation,
            'forward_velocity': self.get_forward_velocity()
        }
        
        self.prev_position = self.robot_position.copy()
        
        return observation, reward, done, info
    
    def apply_action(self, action):
        """Apply torques to robot joints"""
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Publish joint commands (this would need a proper joint controller)
        # For now, we'll use a simple velocity command
        twist_msg = Twist()
        
        # Simple mapping: use action[0] for forward/backward motion
        twist_msg.linear.x = float(action[0] / 100.0)  # Normalize to -1 to 1
        twist_msg.angular.z = float(action[1] / 100.0)  # Use second action for turning
        
        self.cmd_vel_pub.publish(twist_msg)
    
    def get_observation(self):
        """Get current observation"""
        observation = np.concatenate([
            self.joint_positions,
            self.joint_velocities,
            self.robot_position,
            self.robot_orientation
        ])
        return observation.astype(np.float32)
    
    def calculate_reward(self):
        """Calculate reward for current state"""
        reward = 0.0
        
        # Forward velocity reward
        forward_vel = self.get_forward_velocity()
        velocity_reward = -abs(forward_vel - self.target_velocity)
        reward += velocity_reward
        
        # Stability reward (penalize excessive tilting)
        roll, pitch, _ = self.robot_orientation
        stability_reward = -(abs(roll) + abs(pitch)) * 10
        reward += stability_reward
        
        # Height reward (stay upright)
        height_reward = max(0, self.robot_position[2] - 0.5) * 5
        reward += height_reward
        
        # Energy efficiency (penalize large torques)
        # This would need actual torque values from joint states
        
        # Survival bonus
        reward += 1.0
        
        return reward
    
    def get_forward_velocity(self):
        """Calculate forward velocity"""
        dt = 0.02  # Approximate time step
        if dt > 0:
            return (self.robot_position[0] - self.prev_position[0]) / dt
        return 0.0
    
    def is_done(self):
        """Check if episode should terminate"""
        # Terminate if robot falls
        if self.robot_position[2] < 0.3:  # Below certain height
            return True
        
        # Terminate if robot tilts too much
        roll, pitch, _ = self.robot_orientation
        if abs(roll) > 1.0 or abs(pitch) > 1.0:  # More than ~60 degrees
            return True
        
        # Terminate if max steps reached
        if self.current_step >= self.max_episode_steps:
            return True
        
        return False
    
    def close(self):
        """Clean up resources"""
        rclpy.shutdown()