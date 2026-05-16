"""
Launch the policy inference node.

Usage — simulation:
  ros2 launch policy_runner policy_runner.launch.py \
    policy_path:=/path/to/policy_1.pt

Usage — hardware:
  ros2 launch policy_runner policy_runner.launch.py \
    policy_path:=/path/to/policy_1.pt hardware:=true
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory('policy_runner')

    return LaunchDescription([
        DeclareLaunchArgument('policy_path', default_value='',
                              description='Path to JIT-exported .pt policy file'),
        DeclareLaunchArgument('hardware', default_value='false',
                              description='Enable hardware safety guards'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),

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
        ),
    ])
