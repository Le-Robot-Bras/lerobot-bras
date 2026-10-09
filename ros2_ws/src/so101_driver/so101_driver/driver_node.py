#!/usr/bin/env python3
"""driver_node.py - ROS2 driver of the SO-ARM101.

The only node that talks to the backend: SO101Sim (MuJoCo) or SO101Follower
(real arm), chosen by the `use_sim` parameter. The rest of the stack never
knows which one runs.

TODO:
  - Publish /joint_states in RADIANS (>= 20 Hz) (c'est fait)
  - Subscribe to /joint_command (radians, gripper 0-100 %) (c'est fait)
  - Disconnect the backend when the node stops (c'est fait)
"""
from pathlib import Path

import rclpy
import rclpy.node
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from so101_driver.remote_arm import DEFAULT_REMOTE_PORT, SO101Remote
from so101_sim import SO101Sim

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Gripper joint limits in the URDF (rad).
# LeRobot convention: 0 % = closed (lower), 100 % = open (upper).
GRIPPER_RANGE_RAD = (-0.174533, 1.74533)

# Calibration file of the real arm, mounted by docker compose (CALIBRATION_FILE in docker/.env).
CALIBRATION_FILE = Path("/calibration/arm.json")
DEFAULT_PORT = "/dev/ttyACM0"

CONTROL_HZ = 50.0
JOINT_STATE_HZ = 30.0


def _gripper_pct_to_rad(pct: float) -> float:
    """Convert gripper command from 0-100 % to radians."""
    pct = max(0.0, min(100.0, pct))
    return GRIPPER_RANGE_RAD[0] + (GRIPPER_RANGE_RAD[1] - GRIPPER_RANGE_RAD[0]) * (pct / 100.0)


class DriverNode(rclpy.node.Node):
    """Driver SO-ARM101: /joint_command -> backend -> /joint_states."""

    def __init__(self):
        super().__init__("so101_driver")
        self._use_sim: bool = self.declare_parameter("use_sim", True).value
        port: str = self.declare_parameter("port", DEFAULT_PORT).value
        # "host:port" of tools/real_arm_server.py (arm plugged on the host, e.g. macOS)
        self._remote_arm: str = self.declare_parameter("remote_arm", "").value
        # Remote arm only: turn the motors on at startup (the server must allow motion).
        self._enable_torque: bool = self.declare_parameter("enable_torque", False).value
        mode = "simulation" if self._use_sim else "hardware"
        self.get_logger().info(f"Mode: {mode} (port={port}, remote_arm={self._remote_arm or '-'})")

        ##todo ancien
        self._robot = self._connect_arm(port)
        self.get_logger().info(f"Robot connected: {type(self._robot).__name__}")

        self._target: dict[str, float] = {}

        self._pub_states = self.create_publisher(JointState, "/joint_states", 10)
        self.create_subscription(JointState, "/joint_command", self._cb_joint_command, 10)

        self.create_timer(1.0 / CONTROL_HZ, self._control_step)
        self.create_timer(1.0 / JOINT_STATE_HZ, self._publish_joint_states)


        self.get_logger().info("Driver node ready.")

    def _connect_arm(self, port: str = DEFAULT_PORT) -> SO101Sim | SO101Follower | SO101Remote:
        """Creates and connects the backend chosen by use_sim (provided)."""
        if self._use_sim:
            robot = SO101Sim()
            robot.connect()
            return robot

        if self._remote_arm:
            host, _, remote_port = self._remote_arm.partition(":")
            robot = SO101Remote(host, int(remote_port or DEFAULT_REMOTE_PORT))
            robot.connect()
            if self._enable_torque:
                robot.enable_torque()
                self.get_logger().warning("Remote arm: torque ENABLED, the arm will follow /joint_command.")
            else:
                self.get_logger().info("Remote arm: torque off, /joint_command is ignored (enable_torque:=true).")
            return robot

        if not CALIBRATION_FILE.is_file():
            raise RuntimeError(
                f"No calibration file at {CALIBRATION_FILE}: set CALIBRATION_FILE in docker/.env"
            )
        config = SO101FollowerConfig(
            port=port,
            id=CALIBRATION_FILE.stem,
            calibration_dir=CALIBRATION_FILE.parent,
            use_radians=True,
        )
        robot = SO101Follower(config)
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            robot.disconnect()
            raise RuntimeError(
                f"{CALIBRATION_FILE} does not match the motors: wrong file for this arm? "
                "Re-run lerobot-calibrate on the host."
            )
        robot.bus.disable_torque()  # Comment out this line to control the robot.
        self.get_logger().info(
            "Torque disabled: arm can be moved freely by hand. Remove this part to control the arm."
        )
        return robot

    def destroy_node(self):
        if self._robot.is_connected:
            self._robot.disconnect()
            self.get_logger().info("Robot disconnected.")
        super().destroy_node()

    def _cb_joint_command(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._target[f"{name}.pos"] = pos

    def _control_step(self):
        if self._target:
            self._robot.send_action(self._target)

    def _publish_joint_states(self):
        obs = self._robot.get_observation()
        positions = [obs[f"{name}.pos"] for name in JOINT_NAMES]
        positions[-1] = _gripper_pct_to_rad(positions[-1])  # gripper: % -> rad

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = positions
        self._pub_states.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DriverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
