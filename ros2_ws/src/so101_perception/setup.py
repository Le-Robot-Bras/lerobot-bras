from glob import glob

from setuptools import setup

package_name = "so101_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
        entry_points={
        "console_scripts": [
            "perception = so101_perception.perception_node:main",
            "calibrate_camera = so101_perception.calibrate_camera:main",
            "yolo = so101_perception.yolo_node:main",
            "hand = so101_perception.hand_node:main",
            "hand_3d = so101_perception.hand_3d_node:main",
            "aruco = so101_perception.aruco_node:main",
            "hand_debug = so101_perception.hand_debug_node:main",
        ],
    },
)
