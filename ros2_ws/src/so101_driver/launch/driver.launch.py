#!/usr/bin/env python3
"""driver.launch.py - launches the SO-ARM101 driver node

    ros2 launch so101_driver driver.launch.py                # simulation
    ros2 launch so101_driver driver.launch.py use_sim:=false # real robot
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim", default_value="true",
                              description="Simulated backend or real robot"),
        DeclareLaunchArgument("port", default_value="/dev/ttyACM0",  #  Use the following command to find the robot port:
                              # sudo udevadm monitor --subsystem-match=tty --property | grep --line-buffered -E "(DEVNAME=|ACTION=)"
                              description="Serial port of the real SO-ARM101"),
        DeclareLaunchArgument("remote_arm", default_value="",
                              description="host:port of tools/real_arm_server.py (arm on the host, e.g. macOS)"),
        DeclareLaunchArgument("enable_torque", default_value="false",
                              description="Remote arm: enable the motors at startup"),
        Node(
            package="so101_driver",
            executable="driver",
            name="so101_driver",
            output="screen",
            parameters=[{
                "use_sim": ParameterValue(LaunchConfiguration("use_sim"), value_type=bool),
                "port": LaunchConfiguration("port"),
                "remote_arm": LaunchConfiguration("remote_arm"),
                "enable_torque": ParameterValue(LaunchConfiguration("enable_torque"), value_type=bool),
            }],
        ),
    ])
