#!/usr/bin/env python3
"""monitor.py - live, human-readable log of the pick & place (movements + perception).

    ros2 run so101_brain monitor                 # 2 lines/s + one line at each state change
    ros2 run so101_brain monitor --rate 5

Each line:
  time  STATE  | tcp (x, y, z) mm  | gripper %  | cube (x, y) mm yaw deg, age  | box (x, y) mm  | cmd
  - tcp: TCP position from forward kinematics (/end_effector_pose)
  - gripper: measured opening (from /joint_states, converted back to %)
  - cube / box: latest estimate of the perception (/object_position, /drop_box_position);
    age = time since the last detection ("--" = not seen yet). The perception does not
    publish when the cube is hidden (by the arm, in the gripper, in the box), so the age grows.
  - cmd: gripper command sent by the brain (/joint_command)
Read-only: subscribes only, never commands the arm.
"""
import argparse
import math
import time

import rclpy
import rclpy.node
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from std_msgs.msg import String

GRIPPER_RANGE_RAD = (-0.174533, 1.74533)


def _yaw(q) -> float:
    return math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


class Monitor(rclpy.node.Node):
    def __init__(self, rate: float):
        super().__init__("so101_monitor")
        self._state = "?"
        self._status = ""
        self._tcp = None
        self._gripper = None
        self._gripper_cmd = None
        self._cube = None   # (x, y, yaw, t)
        self._box = None
        self._t0 = time.monotonic()
        self.create_subscription(String, "/brain_state", self._cb_state, 10)
        self.create_subscription(PoseStamped, "/end_effector_pose", self._cb_tcp, 10)
        self.create_subscription(JointState, "/joint_states", self._cb_joints, 10)
        self.create_subscription(JointState, "/joint_command", self._cb_command, 10)
        self.create_subscription(PoseStamped, "/object_position", self._cb_cube, 10)
        self.create_subscription(PoseStamped, "/drop_box_position", self._cb_box, 10)
        self.create_timer(1.0 / rate, self._print)

    def _cb_state(self, msg):
        state, _, status = msg.data.partition(" - ")
        changed = state != self._state
        self._state, self._status = state, status
        if changed:
            self._print(event=True)

    def _cb_tcp(self, msg):
        p = msg.pose.position
        self._tcp = (p.x, p.y, p.z)

    def _cb_joints(self, msg):
        if "gripper" in msg.name:
            rad = msg.position[msg.name.index("gripper")]
            lo, hi = GRIPPER_RANGE_RAD
            self._gripper = max(0.0, min(100.0, (rad - lo) / (hi - lo) * 100.0))

    def _cb_command(self, msg):
        if "gripper" in msg.name:
            self._gripper_cmd = msg.position[msg.name.index("gripper")]

    def _cb_cube(self, msg):
        p = msg.pose.position
        self._cube = (p.x, p.y, _yaw(msg.pose.orientation) % 90.0, time.monotonic())

    def _cb_box(self, msg):
        p = msg.pose.position
        self._box = (p.x, p.y, _yaw(msg.pose.orientation) % 90.0, time.monotonic())

    def _print(self, event=False):
        now = time.monotonic()
        tcp = "tcp   --" if self._tcp is None else \
            "tcp ({:+4.0f},{:+4.0f},{:4.0f})mm".format(*(v * 1000 for v in self._tcp))
        grip = "grip  --" if self._gripper is None else f"grip {self._gripper:3.0f}%"
        if self._gripper_cmd is not None:
            grip += f" (cmd {self._gripper_cmd:3.0f}%)"
        cube = "cube --" if self._cube is None else \
            "cube ({:+4.0f},{:+4.0f})mm yaw {:2.0f} age {:4.1f}s".format(
                self._cube[0] * 1000, self._cube[1] * 1000, self._cube[2], now - self._cube[3])
        box = "box --" if self._box is None else \
            "box ({:+4.0f},{:+4.0f})mm".format(self._box[0] * 1000, self._box[1] * 1000)
        mark = ">>" if event else "  "
        line = f"{mark} {self._state:9s} | {tcp} | {grip} | {cube} | {box}"
        if event and self._status:
            line += f"\n   {'':9s}   {self._status}"
        print(line, flush=True)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rate", type=float, default=2.0, help="lines per second (default 2)")
    known, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = Monitor(known.rate)
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
