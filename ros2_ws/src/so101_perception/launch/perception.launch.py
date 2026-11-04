#!/usr/bin/env python3
"""perception.launch.py - red cube + drop box detection in the robot frame.

    ros2 launch so101_perception perception.launch.py                        # sim: camera pose from the driver
    ros2 launch so101_perception perception.launch.py calibration_file:=/calibration/camera.json   # real camera
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("calibration_file", default_value="",
                              description="JSON from calibrate_camera (K, D, T camera->robot)"),
        DeclareLaunchArgument("publish_hz", default_value="10.0"),
        Node(
            package="so101_perception",
            executable="perception",
            name="so101_perception",
            output="screen",
            parameters=[{
                "calibration_file": ParameterValue(LaunchConfiguration("calibration_file"), value_type=str),
                "publish_hz": ParameterValue(LaunchConfiguration("publish_hz"), value_type=float),
            }],
        ),
    ])
