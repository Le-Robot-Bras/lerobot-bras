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
        DeclareLaunchArgument("use_real_camera", default_value="false",
                              description="Use real physical camera or sim camera"),
        DeclareLaunchArgument("camera_id", default_value="0",
                              description="Webcam index (e.g. 0 for /dev/video0)"),
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
                "use_real_camera": ParameterValue(LaunchConfiguration("use_real_camera"), value_type=bool),
                "camera_id": ParameterValue(LaunchConfiguration("camera_id"), value_type=int),
                "port": LaunchConfiguration("port"),
                "remote_arm": LaunchConfiguration("remote_arm"),
                "enable_torque": ParameterValue(LaunchConfiguration("enable_torque"), value_type=bool),
            }],
        ),
    ])
