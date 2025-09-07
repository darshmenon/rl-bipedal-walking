import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_bipedal_description = get_package_share_directory('bipedal_robot_description')

    # Declare launch arguments
    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value=os.path.join(pkg_bipedal_description, 'worlds', 'empty_world.sdf'),
        description='World file to load in Ignition Gazebo'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
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

    # Launch Ignition Gazebo
    gz_sim = Node(
        package='ros_gz_sim',
        executable='gz_sim',
        output='screen',
        arguments=[
            LaunchConfiguration('world_file'),
            '-v', '4'   # verbosity
        ]
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
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    return LaunchDescription([
        world_file_arg,
        use_sim_time_arg,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        joint_state_publisher_gui
    ])
