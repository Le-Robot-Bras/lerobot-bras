from glob import glob

from setuptools import setup

package_name = "so101_driver"

setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    zip_safe=True,
    description="ROS2 driver of the SO-ARM101 (MuJoCo sim or real arm).",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "driver = so101_driver.driver_node:main",
            "dance_bridge = so101_driver.dance_bridge:main",
        ],
    },
)
