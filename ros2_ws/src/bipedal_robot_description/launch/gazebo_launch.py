import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_bipedal_description = get_package_share_directory('bipedal_robot_description')

    # Declare launch arguments
    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value='empty.sdf',
        description='World file or Gazebo resource name to load in Gazebo Sim'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    joint_state_gui_arg = DeclareLaunchArgument(
        'joint_state_gui',
        default_value='false',
        description='Launch joint_state_publisher_gui for manual joint control'
    )

    # Load robot description
    urdf_file = os.path.join(pkg_bipedal_description, 'urdf', 'bipedal.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )

    # Launch Gazebo Sim through the packaged ros_gz_sim launch file.
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_args': ['-r ', LaunchConfiguration('world_file')]
        }.items()
    )

    # Spawn robot in Ignition Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-string', robot_description,
            '-name', 'bipedal_robot',
            '-allow_renaming', 'true'
        ],
        output='screen'
    )

    # Optional: Joint state publisher GUI
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('joint_state_gui'))
    )

    return LaunchDescription([
        world_file_arg,
        use_sim_time_arg,
        joint_state_gui_arg,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        joint_state_publisher_gui
    ])
