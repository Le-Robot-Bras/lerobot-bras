#!/usr/bin/env python3
"""perception.launch.py - red cube + drop box detection in the robot frame.

    ros2 launch so101_perception perception.launch.py                        # sim: camera pose from the driver
    ros2 launch so101_perception perception.launch.py calibration_file:=/calibration/camera.json   # real camera
    ros2 launch so101_perception perception.launch.py detector:=yolo                 # YOLO-seg masks
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
        DeclareLaunchArgument("detector", default_value="hsv", description="hsv | yolo"),
        DeclareLaunchArgument("yolo_weights", default_value="",
                              description="YOLO-seg weights (default: the package models/so101_seg.pt)"),
        Node(
            package="so101_perception",
            executable="perception",
            name="so101_perception",
            output="screen",
            parameters=[{
                "calibration_file": ParameterValue(LaunchConfiguration("calibration_file"), value_type=str),
                "publish_hz": ParameterValue(LaunchConfiguration("publish_hz"), value_type=float),
                "detector": ParameterValue(LaunchConfiguration("detector"), value_type=str),
                "yolo_weights": ParameterValue(LaunchConfiguration("yolo_weights"), value_type=str),
            }],
        ),
    ])
