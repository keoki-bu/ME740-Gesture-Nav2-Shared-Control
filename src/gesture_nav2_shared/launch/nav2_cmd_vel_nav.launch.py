import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import SetRemap
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    map_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')

    turtlebot3_nav_launch = os.path.join(
        get_package_share_directory('turtlebot3_navigation2'),
        'launch',
        'navigation2.launch.py'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=os.path.expanduser('~/maps/tb3_map.yaml'),
            description='Full path to map yaml file'
        ),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation clock'
        ),

        GroupAction([
            # Nav2 controller raw output
            SetRemap(src='/cmd_vel', dst='/cmd_vel_nav_raw'),
            SetRemap(src='cmd_vel', dst='/cmd_vel_nav_raw'),

            # Nav2 velocity_smoother final output
            # This is the velocity that fusion should read.
            SetRemap(src='/cmd_vel_smoothed', dst='/cmd_vel_nav'),
            SetRemap(src='cmd_vel_smoothed', dst='/cmd_vel_nav'),

            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(turtlebot3_nav_launch),
                launch_arguments={
                    'map': map_file,
                    'use_sim_time': use_sim_time,
                }.items()
            ),
        ])
    ])
