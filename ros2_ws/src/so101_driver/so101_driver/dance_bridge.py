"""dance_bridge.py - shows the live dance in ROS 2 / RViz.

The dance (tools/dance_live.py --ros-bridge, on the host) owns the arm link (real_arm_server accepts
one client), so this node does not talk to the arm: it reads the dance's state stream (JSON lines on
TCP, default host.docker.internal:5600) and republishes it:
  /joint_states  (sensor_msgs/JointState): the MEASURED real arm if available, else the target
                 -> robot_state_publisher -> TF -> RViz shows the arm dancing
  /dance/target  (sensor_msgs/JointState): the commanded pose
  /dance/state   (std_msgs/String): "<style> | <genre> | <move/beat label>"

    ros2 run so101_driver dance_bridge --ros-args -p host:=host.docker.internal -p port:=5600

Do not run the driver node at the same time (it also publishes /joint_states).
"""
import json
import socket
import threading
import time

import rclpy
import rclpy.node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from std_msgs.msg import String

GRIPPER_RANGE_RAD = (-0.174533, 1.74533)  # URDF; LeRobot 0 % = closed, 100 % = open


def gripper_pct_to_rad(pct):
    pct = max(0.0, min(100.0, pct))
    return GRIPPER_RANGE_RAD[0] + (GRIPPER_RANGE_RAD[1] - GRIPPER_RANGE_RAD[0]) * pct / 100.0


def to_joint_state(stamp, names, values):
    msg = JointState()
    msg.header.stamp = stamp
    msg.name = list(names)
    msg.position = [gripper_pct_to_rad(v) if n == "gripper" else float(v) for n, v in zip(names, values)]
    return msg


class DanceBridge(rclpy.node.Node):
    def __init__(self):
        super().__init__("dance_bridge")
        self.host = self.declare_parameter("host", "host.docker.internal").value
        self.port = int(self.declare_parameter("port", 5600).value)
        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_target = self.create_publisher(JointState, "/dance/target", 10)
        self.pub_state = self.create_publisher(String, "/dance/state", 10)
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        self.get_logger().info(f"waiting for the dance stream on {self.host}:{self.port} "
                               "(tools/dance_live.py --ros-bridge on the host)")

    def _loop(self):
        while not self._stop.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=2.0) as s:
                    self.get_logger().info("connected to the dance stream")
                    s.settimeout(5.0)
                    for line in s.makefile("r"):
                        if self._stop.is_set():
                            return
                        self._publish(json.loads(line))
            except (OSError, ValueError) as e:
                self.get_logger().warn(f"dance stream unavailable ({e}); retrying in 1 s", throttle_duration_sec=10.0)
                time.sleep(1.0)

    def _publish(self, d):
        stamp = self.get_clock().now().to_msg()
        names = d["names"]
        shown = d.get("measured") or d["target"]
        self.pub_js.publish(to_joint_state(stamp, names, shown))
        self.pub_target.publish(to_joint_state(stamp, names, d["target"]))
        self.pub_state.publish(String(data=f"{d.get('style', '')} | {d.get('genre', '')} | {d.get('label', '')}"))

    def destroy_node(self):
        self._stop.set()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DanceBridge()
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
