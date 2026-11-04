#!/usr/bin/env python3
"""driver_node.py - ROS2 driver of the SO-ARM101.

The only node that talks to the backend: SO101Sim (MuJoCo), SO101Follower (real
arm on a serial port) or SO101Remote (real arm served from the host, macOS),
chosen by the `use_sim` / `remote_arm` parameters. The rest of the stack never
knows which one runs.

Topics:
  /joint_states               (pub, 30 Hz)  radians, gripper converted to radians (URDF joint)
  /joint_command              (sub)         radians, gripper 0-100 %; partial commands allowed,
                                            every value clamped to the joint limits
  /external_cam/image_raw     (pub, 10 Hz)  sim: MuJoCo render; real: `camera_device` (OpenCV)
  /external_cam/camera_info   (pub, 10 Hz)  intrinsics K
  /sim/reset_object           (srv, std_srvs/Trigger, sim only) put the cube at a new random
                                            reachable pose (demos: one pick & place per call)
Parameters published for the perception (sim only): cam_K (9), cam_T (16, camera -> world,
MuJoCo camera axes). This is the ground-truth pose of the sim camera; on the real robot
the perception uses its own calibration file instead (calibrate_camera).
"""
from pathlib import Path

import numpy as np
import rclpy
import rclpy.node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Trigger

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Joint limits (rad), from the URDF / MJCF. Gripper commands are in % (0-100).
JOINT_LIMITS = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (0.0, 100.0),
}

# Gripper joint limits in the URDF (rad).
# LeRobot convention: 0 % = closed (lower), 100 % = open (upper).
GRIPPER_RANGE_RAD = (-0.174533, 1.74533)

# Calibration file of the real arm, mounted by docker compose (CALIBRATION_FILE in docker/.env).
CALIBRATION_FILE = Path("/calibration/arm.json")
DEFAULT_PORT = "/dev/ttyACM0"

CONTROL_HZ = 50.0
JOINT_STATE_HZ = 30.0
CAMERA_HZ = 10.0
CAMERA_FRAME = "external_cam"


def _gripper_pct_to_rad(pct: float) -> float:
    """Convert gripper opening from 0-100 % to radians."""
    pct = max(0.0, min(100.0, pct))
    return GRIPPER_RANGE_RAD[0] + (GRIPPER_RANGE_RAD[1] - GRIPPER_RANGE_RAD[0]) * (pct / 100.0)


class DriverNode(rclpy.node.Node):
    """Driver SO-ARM101: /joint_command -> backend -> /joint_states (+ camera)."""

    def __init__(self):
        super().__init__("so101_driver")
        self._use_sim: bool = self.declare_parameter("use_sim", True).value
        port: str = self.declare_parameter("port", DEFAULT_PORT).value
        # "host:port" of tools/real_arm_server.py (arm plugged on the host, e.g. macOS)
        self._remote_arm: str = self.declare_parameter("remote_arm", "").value
        # Real arm: turn the motors on at startup. Off by default: the arm stays limp
        # and /joint_command is ignored (safety).
        self._enable_torque: bool = self.declare_parameter("enable_torque", False).value
        # Real camera (index or device path) when not in simulation; "" = no camera.
        camera_device: str = str(self.declare_parameter("camera_device", "").value)
        mode = "simulation" if self._use_sim else "hardware"
        self.get_logger().info(f"Mode: {mode} (port={port}, remote_arm={self._remote_arm or '-'})")

        self._robot = self._connect_arm(port)
        self.get_logger().info(f"Robot connected: {type(self._robot).__name__}")

        self._target: dict[str, float] = {}

        self._pub_states = self.create_publisher(JointState, "/joint_states", 10)
        self.create_subscription(JointState, "/joint_command", self._cb_joint_command, 10)
        self.create_timer(1.0 / CONTROL_HZ, self._control_step)
        self.create_timer(1.0 / JOINT_STATE_HZ, self._publish_joint_states)
        self.create_service(Trigger, "/sim/reset_object", self._cb_reset_object)

        self._camera = None
        self._cam_K = None
        if self._use_sim:
            self._setup_sim_camera()
        elif camera_device:
            self._setup_real_camera(camera_device)
        if self._camera is not None:
            self._pub_image = self.create_publisher(Image, "/external_cam/image_raw", qos_profile_sensor_data)
            self._pub_info = self.create_publisher(CameraInfo, "/external_cam/camera_info", qos_profile_sensor_data)
            self.create_timer(1.0 / CAMERA_HZ, self._publish_camera)

        self.get_logger().info("Driver node ready.")

    # ------------------------------------------------------------- backends

    def _connect_arm(self, port: str = DEFAULT_PORT):
        """Creates and connects the backend chosen by use_sim / remote_arm."""
        if self._use_sim:
            from so101_sim import SO101Sim
            robot = SO101Sim()
            robot.connect()
            return robot

        if self._remote_arm:
            from so101_driver.remote_arm import DEFAULT_REMOTE_PORT, SO101Remote
            host, _, remote_port = self._remote_arm.partition(":")
            robot = SO101Remote(host, int(remote_port or DEFAULT_REMOTE_PORT))
            robot.connect()
            if self._enable_torque:
                robot.enable_torque()
                self.get_logger().warning("Remote arm: torque ENABLED, the arm will follow /joint_command.")
            else:
                self.get_logger().info("Remote arm: torque off, /joint_command is ignored (enable_torque:=true).")
            return robot

        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
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
        if self._enable_torque:
            self.get_logger().warning("Real arm: torque ENABLED, the arm will follow /joint_command.")
        else:
            robot.bus.disable_torque()
            self.get_logger().info(
                "Real arm: torque disabled, the arm can be moved by hand and /joint_command is "
                "ignored. Relaunch with enable_torque:=true (ARM_TORQUE=true in docker/.env) to control it."
            )
        return robot

    def destroy_node(self):
        """Disconnects the backend. On the real arm this disables the torque."""
        if self._camera is not None and not self._use_sim:
            self._camera.release()
        if self._robot.is_connected:
            self._robot.disconnect()
            self.get_logger().info("Robot disconnected.")
        super().destroy_node()

    def _cb_reset_object(self, request, response):
        """Simulation only: new random cube pose (SO101Sim.randomize_object)."""
        if not self._use_sim:
            response.success, response.message = False, "only available in simulation (use_sim:=true)"
            return response
        self._robot.randomize_object()
        response.success = True
        response.message = "cube moved to a new random pose"
        self.get_logger().info("/sim/reset_object: cube moved")
        return response

    # ---------------------------------------------------------------- joints

    def _cb_joint_command(self, msg: JointState):
        if len(msg.name) != len(msg.position):
            self.get_logger().warning("Ignoring /joint_command: name and position sizes differ")
            return
        for name, pos in zip(msg.name, msg.position):
            if name not in JOINT_LIMITS:
                self.get_logger().warning(f"Ignoring unknown joint '{name}'", throttle_duration_sec=5.0)
                continue
            if not np.isfinite(pos):
                self.get_logger().warning(f"Ignoring non-finite command for '{name}'")
                continue
            lo, hi = JOINT_LIMITS[name]
            self._target[f"{name}.pos"] = float(min(max(pos, lo), hi))

    def _control_step(self):
        if not self._target:
            return
        if not self._use_sim and not self._enable_torque:
            return  # torque off: do not send goals to a limp arm
        self._robot.send_action(self._target)

    def _publish_joint_states(self):
        obs = self._robot.get_observation()
        positions = [float(obs[f"{name}.pos"]) for name in JOINT_NAMES]
        positions[-1] = _gripper_pct_to_rad(positions[-1])  # gripper: % -> rad

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = positions
        self._pub_states.publish(msg)

    # ---------------------------------------------------------------- camera

    def _setup_sim_camera(self):
        self._camera = "sim"
        K = np.array(self._robot.get_camera_intrinsics(), dtype=float)
        # MuJoCo puts the optical centre at W/2 in continuous coordinates; OpenCV
        # puts pixel centres on integers: cx = W/2 - 0.5 (measured: removes a ~4 mm bias).
        K[0, 2] -= 0.5
        K[1, 2] -= 0.5
        self._cam_K = K
        self._cam_size = (self._robot.camera_width, self._robot.camera_height)
        T = np.array(self._robot.get_camera_extrinsics(), dtype=float)
        self.declare_parameter("cam_K", K.ravel().tolist())
        self.declare_parameter("cam_T", T.ravel().tolist())

    def _setup_real_camera(self, device: str):
        import cv2
        cap = cv2.VideoCapture(int(device) if device.isdigit() else device)
        if not cap.isOpened():
            self.get_logger().error(f"Cannot open camera '{device}': no image will be published")
            return
        self._camera = cap
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cam_size = (w, h)
        # Rough pinhole guess (60 deg horizontal FOV): the perception uses its own
        # calibration file (calibrate_camera) on the real robot.
        f = w / (2 * np.tan(np.radians(30)))
        self._cam_K = np.array([[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1]])
        self.get_logger().info(f"Camera '{device}' opened ({w}x{h})")

    def _publish_camera(self):
        if self._camera == "sim":
            rgb = self._robot.render_camera()
            encoding = "rgb8"
        else:
            ok, bgr = self._camera.read()
            if not ok:
                self.get_logger().warning("Camera read failed", throttle_duration_sec=5.0)
                return
            rgb, encoding = bgr, "bgr8"
        stamp = self.get_clock().now().to_msg()

        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = CAMERA_FRAME
        img.height, img.width = rgb.shape[:2]
        img.encoding = encoding
        img.step = img.width * 3
        img.data = np.ascontiguousarray(rgb).tobytes()
        self._pub_image.publish(img)

        info = CameraInfo()
        info.header = img.header
        info.width, info.height = img.width, img.height
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = self._cam_K.ravel().tolist()
        info.r = np.eye(3).ravel().tolist()
        P = np.zeros((3, 4))
        P[:, :3] = self._cam_K
        info.p = P.ravel().tolist()
        self._pub_info.publish(info)


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
