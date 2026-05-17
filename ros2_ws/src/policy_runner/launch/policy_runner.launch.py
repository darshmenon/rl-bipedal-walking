"""
Launch the policy inference node.

Usage — simulation (XBot-L):
  ros2 launch policy_runner policy_runner.launch.py \
    policy_path:=/path/to/policy.pt

Usage — simulation (H1):
  ros2 launch policy_runner policy_runner.launch.py \
    policy_path:=/path/to/policy.pt robot:=h1

Usage — hardware:
  ros2 launch policy_runner policy_runner.launch.py \
    policy_path:=/path/to/policy.pt robot:=h1 hardware:=true
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory('policy_runner')

    robot = LaunchConfiguration('robot')
    is_h1 = PythonExpression(["'", robot, "' == 'h1'"])

    return LaunchDescription([
        DeclareLaunchArgument('policy_path', default_value='',
                              description='Path to JIT-exported .pt policy file'),
        DeclareLaunchArgument('hardware', default_value='false',
                              description='Enable hardware safety guards'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('robot', default_value='xbot',
                              description='Robot config: xbot or h1'),

        # XBot-L node
        Node(
            package='policy_runner',
            executable='policy_inference',
            name='policy_inference',
            parameters=[
                os.path.join(pkg, 'config', 'xbot_joints.yaml'),
                {
                    'policy_path': LaunchConfiguration('policy_path'),
                    'hardware':    LaunchConfiguration('hardware'),
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                },
            ],
            output='screen',
            condition=UnlessCondition(is_h1),
        ),

        # H1 node
        Node(
            package='policy_runner',
            executable='policy_inference',
            name='policy_inference',
            parameters=[
                os.path.join(pkg, 'config', 'h1_joints.yaml'),
                {
                    'policy_path': LaunchConfiguration('policy_path'),
                    'hardware':    LaunchConfiguration('hardware'),
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                },
            ],
            output='screen',
            condition=IfCondition(is_h1),
        ),
    ])
