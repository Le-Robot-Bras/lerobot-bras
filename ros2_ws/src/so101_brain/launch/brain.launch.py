#!/usr/bin/env python3
"""brain.launch.py - launches the brain of the arm (IK + pick & place).

    ros2 launch so101_brain brain.launch.py                   # then: ros2 service call /brain/start std_srvs/srv/Trigger
    ros2 launch so101_brain brain.launch.py auto_start:=true  # starts as soon as /joint_states arrive
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("auto_start", default_value="false"),
        Node(
            package="so101_brain",
            executable="brain",
            name="so101_brain",
            output="screen",
            parameters=[{"auto_start": ParameterValue(LaunchConfiguration("auto_start"), value_type=bool)}],
        ),
    ])
