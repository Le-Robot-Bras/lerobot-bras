from glob import glob

from setuptools import setup

package_name = "so101_brain"

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
    description="IK + pick & place loop of the SO-ARM101.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "brain = so101_brain.brain_node:main",
            "demo_loop = so101_brain.demo_loop:main",
            "monitor = so101_brain.monitor:main",
        ],
    },
)
