#!/usr/bin/env python3
"""
policy_inference_node.py

Loads a JIT-exported locomotion policy (.pt) and runs it in a 100 Hz loop.
Subscribes to:
  /joint_states          (sensor_msgs/JointState)
  /odom                  (nav_msgs/Odometry)   — base velocity
  /cmd_vel               (geometry_msgs/Twist) — velocity command

Publishes to:
  /position_controller/commands  (std_msgs/Float64MultiArray)
  — compatible with ros2_control ForwardCommandController

Designed for both Gazebo Sim validation and real hardware deployment.
Set the 'hardware' parameter to true to enable hardware-specific safety guards.
"""

import math
import collections
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class PolicyInferenceNode(Node):

    def __init__(self):
        super().__init__('policy_inference')

        # ── parameters ──────────────────────────────────────────────────
        self.declare_parameter('policy_path', '')
        self.declare_parameter('joint_names', [
            'left_leg_roll_joint', 'left_leg_yaw_joint', 'left_leg_pitch_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_leg_roll_joint', 'right_leg_yaw_joint', 'right_leg_pitch_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        ])
        self.declare_parameter('joint_command_topic', '/position_controller/commands')
        self.declare_parameter('policy_frequency', 100.0)
        self.declare_parameter('obs_scale_lin_vel', 2.0)
        self.declare_parameter('obs_scale_ang_vel', 1.0)
        self.declare_parameter('obs_scale_dof_pos', 1.0)
        self.declare_parameter('obs_scale_dof_vel', 0.05)
        self.declare_parameter('obs_scale_quat',    1.0)
        self.declare_parameter('frame_stack',     15)
        self.declare_parameter('num_single_obs',  47)
        self.declare_parameter('clip_obs',        18.0)
        self.declare_parameter('clip_actions',    18.0)
        self.declare_parameter('default_joint_pos', [0.0] * 12)
        self.declare_parameter('hardware', False)
        self.declare_parameter('action_scale', 0.25)

        policy_path       = self.get_parameter('policy_path').value
        self.joint_names  = self.get_parameter('joint_names').value
        cmd_topic         = self.get_parameter('joint_command_topic').value
        freq              = self.get_parameter('policy_frequency').value
        self.frame_stack  = self.get_parameter('frame_stack').value
        self.num_single_obs = self.get_parameter('num_single_obs').value
        self.clip_obs     = self.get_parameter('clip_obs').value
        self.clip_actions = self.get_parameter('clip_actions').value
        self.default_pos  = np.array(self.get_parameter('default_joint_pos').value, dtype=np.float32)
        self.hardware     = self.get_parameter('hardware').value
        self.action_scale = self.get_parameter('action_scale').value

        self.obs_scales = {
            'lin_vel': self.get_parameter('obs_scale_lin_vel').value,
            'ang_vel': self.get_parameter('obs_scale_ang_vel').value,
            'dof_pos': self.get_parameter('obs_scale_dof_pos').value,
            'dof_vel': self.get_parameter('obs_scale_dof_vel').value,
            'quat':    self.get_parameter('obs_scale_quat').value,
        }

        self.num_joints   = len(self.joint_names)

        # ── load policy ─────────────────────────────────────────────────
        if not policy_path:
            self.get_logger().error('policy_path parameter is empty — set it to a .pt file')
            raise RuntimeError('No policy path provided')
        self.policy = torch.jit.load(policy_path, map_location='cpu')
        self.policy.eval()
        self.get_logger().info(f'Loaded policy from {policy_path}')

        # ── state ────────────────────────────────────────────────────────
        self.dof_pos      = np.zeros(self.num_joints, dtype=np.float32)
        self.dof_vel      = np.zeros(self.num_joints, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.base_euler   = np.zeros(3, dtype=np.float32)  # roll, pitch, yaw
        self.commands     = np.zeros(3, dtype=np.float32)  # vx, vy, yaw_rate
        self.last_actions = np.zeros(self.num_joints, dtype=np.float32)
        self.episode_step = 0

        # observation history for frame stacking
        self.obs_history = collections.deque(
            [np.zeros(self.num_single_obs, dtype=np.float32)] * self.frame_stack,
            maxlen=self.frame_stack,
        )

        # ── pub/sub ──────────────────────────────────────────────────────
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, qos)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, qos)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_cb, 10)

        self.cmd_pub = self.create_publisher(Float64MultiArray, cmd_topic, 10)

        self.create_timer(1.0 / freq, self._policy_step)
        self.get_logger().info(
            f'Policy runner ready at {freq:.0f} Hz — '
            f'{"HARDWARE mode" if self.hardware else "simulation mode"}'
        )

    # ── callbacks ────────────────────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState):
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.dof_pos[i] = msg.position[idx] if msg.position else 0.0
                self.dof_vel[i] = msg.velocity[idx]  if msg.velocity  else 0.0

    def _odom_cb(self, msg: Odometry):
        self.base_lin_vel[0] = msg.twist.twist.linear.x
        self.base_lin_vel[1] = msg.twist.twist.linear.y
        self.base_lin_vel[2] = msg.twist.twist.linear.z
        self.base_ang_vel[0] = msg.twist.twist.angular.x
        self.base_ang_vel[1] = msg.twist.twist.angular.y
        self.base_ang_vel[2] = msg.twist.twist.angular.z
        q = msg.pose.pose.orientation
        self.base_euler = self._quat_to_euler(q.x, q.y, q.z, q.w)

    def _cmd_vel_cb(self, msg: Twist):
        self.commands[0] = msg.linear.x
        self.commands[1] = msg.linear.y
        self.commands[2] = msg.angular.z

    # ── main loop ────────────────────────────────────────────────────────

    def _policy_step(self):
        obs = self._build_single_obs()
        obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        self.obs_history.append(obs)

        # stack frames: (frame_stack, num_single_obs) → flat
        stacked = np.stack(list(self.obs_history), axis=0).reshape(-1)
        obs_tensor = torch.from_numpy(stacked).unsqueeze(0)

        with torch.no_grad():
            actions = self.policy(obs_tensor).squeeze(0).numpy()

        actions = np.clip(actions, -self.clip_actions, self.clip_actions)
        targets = self.action_scale * actions + self.default_pos

        if self.hardware:
            targets = self._apply_safety_limits(targets)

        self.last_actions = actions.copy()
        self.episode_step += 1

        msg = Float64MultiArray()
        msg.data = targets.tolist()
        self.cmd_pub.publish(msg)

    # ── helpers ──────────────────────────────────────────────────────────

    def _build_single_obs(self):
        t = self.episode_step * 0.01          # 100 Hz → dt=0.01 s
        cycle_time = 0.64
        phase = t / cycle_time
        sin_p = math.sin(2 * math.pi * phase)
        cos_p = math.cos(2 * math.pi * phase)

        cmd = self.commands * np.array([
            self.obs_scales['lin_vel'],
            self.obs_scales['lin_vel'],
            self.obs_scales['ang_vel'],
        ])

        q  = (self.dof_pos - self.default_pos) * self.obs_scales['dof_pos']
        dq = self.dof_vel * self.obs_scales['dof_vel']

        ang = self.base_ang_vel * self.obs_scales['ang_vel']
        eul = self.base_euler   * self.obs_scales['quat']

        obs = np.concatenate([
            [sin_p, cos_p], cmd,        # 5
            q,                          # num_joints
            dq,                         # num_joints
            self.last_actions,          # num_joints
            ang,                        # 3
            eul[:2],                    # 2  (roll, pitch only)
            [eul[2]],                   # 1  yaw
        ])
        return obs.astype(np.float32)

    @staticmethod
    def _quat_to_euler(x, y, z, w):
        """Returns (roll, pitch, yaw) in radians."""
        roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = math.asin( max(-1.0, min(1.0, 2*(w*y - z*x))))
        yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _apply_safety_limits(self, targets: np.ndarray) -> np.ndarray:
        """Hardware-mode guard: limit joint position step size per cycle."""
        max_delta = 0.1  # rad per control step
        delta = targets - (self.action_scale * self.last_actions + self.default_pos)
        delta = np.clip(delta, -max_delta, max_delta)
        return self.action_scale * self.last_actions + self.default_pos + delta


def main(args=None):
    rclpy.init(args=args)
    node = PolicyInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
