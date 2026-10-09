#!/usr/bin/env python3
"""Red cube and brown box detection, projected onto known horizontal planes."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import CameraInfo, Image


# -- HSV ranges for the red cube -------------------------------------------------
# Measured on the MuJoCo scene (OpenCV HSV: H in 0-179): the cube is H ~168.
# The low-hue red range (H 0-10) only catches the brown drop box, so it is unused.
# Tune according to the lighting of the scene (and of the real camera).
HSV_MIN = np.array([160, 120, 70], dtype=np.uint8)
HSV_MAX = np.array([180, 255, 255], dtype=np.uint8)

# The cube has holes: close them so the mask is one blob, ignore specks below this.
MIN_BLOB_AREA_PX = 30
CLOSE_KERNEL = np.ones((5, 5), np.uint8)

# Height (m) of the tracked point: centre of the 5 cm cube resting on the table.
TRACKED_Z = 0.025

# cam_T comes from MuJoCo (x right, y up, z backwards); the pinhole ray below is
# OpenCV (x right, y down, z forward). This flips y and z.
MUJOCO_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


class PerceptionNode(rclpy.node.Node):
    """Publish image-derived positions in the calibrated robot/world frame."""

    def __init__(self):
        super().__init__("so101_perception")

        # Convert incoming ROS images to OpenCV images.
        self.bridge = CvBridge()

        # Unknown until camera calibration is received; never assume identity.
        self.cam_K = None
        self.cam_T = None
        self.object_position = None
        self.drop_box_position = None
        self.cam_D = np.zeros(5)
        self._camera_size = None
        self._image_stamp = None
        self._last_publish_time = None
        publish_hz = float(self.declare_parameter("publish_hz", 10.0).value)
        if not np.isfinite(publish_hz) or publish_hz <= 0:
            raise ValueError("publish_hz must be positive")
        self._publish_period = 1.0 / publish_hz
        self.camera_axes = self.declare_parameter("camera_axes", "mujoco").value
        if self.camera_axes not in ("mujoco", "opencv"):
            raise ValueError("camera_axes must be mujoco or opencv")
        self.output_frame = self.declare_parameter("output_frame", "world").value
        self.object_z = float(self.declare_parameter("object_z", TRACKED_Z).value)
        self.box_z = float(self.declare_parameter("box_z", 0.025).value)
        # Current scene exposes mainly the camera-facing y wall; its centre
        # is half a box width away from the desired drop point.
        self.box_half_width = float(self.declare_parameter("box_half_width", 0.05).value)
        self.box_visible_wall = self.declare_parameter("box_visible_wall", "y").value
        if not all(np.isfinite(v) for v in (self.object_z, self.box_z, self.box_half_width)) or self.box_half_width < 0:
            raise ValueError("Plane heights must be finite and box_half_width nonnegative")
        if self.box_visible_wall not in ("x", "y", "none"):
            raise ValueError("box_visible_wall must be x, y or none")
        calibration_file = self.declare_parameter("calibration_file", "").value
        if calibration_file:
            self._load_calibration(calibration_file)
        self._cam_params_future = None
        self._cam_params_client = self.create_client(
            GetParameters, "/so101_driver/get_parameters"
        )
        self._cam_params_timer = self.create_timer(1.0, self._fetch_cam_params)

        self._pub_object = self.create_publisher(
            PoseStamped, "/object_position", 10
        )
        self._pub_drop_box = self.create_publisher(
            PoseStamped, "/drop_box_position", 10
        )

        # Best-effort subscribers also accept best-effort camera publishers.
        self._sub_camera_info = self.create_subscription(
            CameraInfo,
            "/external_cam/camera_info",
            self._cb_camera_info,
            qos_profile_sensor_data,
        )
        self._sub_image = self.create_subscription(
            Image,
            "/external_cam/image_raw",
            self._cb_image,
            qos_profile_sensor_data,
        )

        self.get_logger().info("Perception ready; waiting for calibrated camera images")

    def _cb_camera_info(self, msg):
        try:
            K = np.asarray(msg.k, dtype=float).reshape(3, 3)
            self._validate_K(K)
            D = np.asarray(msg.d, dtype=float)
            if msg.distortion_model not in ("", "plumb_bob", "rational_polynomial"):
                raise ValueError("unsupported distortion model")
            if D.size not in (0, 4, 5, 8, 12, 14) or not np.all(np.isfinite(D)):
                raise ValueError("invalid distortion coefficients")
            if msg.width <= 0 or msg.height <= 0:
                raise ValueError("invalid image dimensions")
        except ValueError as exc:
            self.get_logger().warning(f"Ignoring CameraInfo: {exc}")
            return
        self.cam_K = K
        self.cam_D = D
        self._camera_size = (msg.width, msg.height)

    def _cb_image(self, msg):
        # Reset every frame; never keep a position after detection is lost.
        self.object_position = None
        self.drop_box_position = None
        if self.cam_K is None or self.cam_T is None:
            return
        if self._camera_size and (msg.width, msg.height) != self._camera_size:
            self.get_logger().warning("Image size differs from camera calibration")
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            center, _ = self._detect_object(image)
            if center is not None:
                self.object_position = self._project_to_3d(*center)
            center, _ = self._detect_box(image)
            if center is not None:
                self.drop_box_position = self._project_box(*center)
        except (CvBridgeError, cv2.error, ValueError) as exc:
            self.object_position = None
            self.drop_box_position = None
            self.get_logger().warning(f"Image processing failed: {exc}")
            return
        self._image_stamp = msg.header.stamp
        now = time.monotonic()
        if self._last_publish_time is None or now - self._last_publish_time >= self._publish_period:
            self._publish_result()
            self._last_publish_time = now

    def _detect_object(self, cv_image):
        """Finds the red cube in a BGR image.

        Returns ((u, v), mask): the pixel center of the cube and the binary mask,
        or (None, mask) if nothing big enough is visible.
        """
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_MIN, HSV_MAX)
        # Real red can wrap around H=0; exclude low-saturation brown/grey.
        mask |= cv2.inRange(hsv, np.array([0, 190, 70]), np.array([8, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask
        blob = max(contours, key=cv2.contourArea)
        if cv2.contourArea(blob) < MIN_BLOB_AREA_PX:
            return None, mask

        m = cv2.moments(blob)
        if m["m00"] == 0:
            return None, mask
        return (m["m10"] / m["m00"], m["m01"] / m["m00"]), mask

    def _detect_box(self, cv_image):
        """Brown box centre from its silhouette (approximate under occlusion)."""
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([5, 80, 25]), np.array([25, 230, 220]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask
        blob = max(contours, key=cv2.contourArea)
        if cv2.contourArea(blob) < MIN_BLOB_AREA_PX:
            return None, mask
        # The projected centre of a planar rectangle is the intersection of
        # its diagonals, rather than the centroid of its perspective silhouette.
        hull = cv2.convexHull(blob)
        corners = cv2.approxPolyDP(hull, .03 * cv2.arcLength(hull, True), True)
        if len(corners) == 4:
            a, b, c, d = corners[:, 0, :].astype(float)
            matrix = np.column_stack((c - a, -(d - b)))
            if abs(np.linalg.det(matrix)) > 1e-6:
                t = np.linalg.solve(matrix, b - a)[0]
                return tuple(a + t * (c - a)), mask
        moment = cv2.moments(blob)
        if moment["m00"] == 0:
            return None, mask
        return (moment["m10"] / moment["m00"], moment["m01"] / moment["m00"]), mask

    def _project_to_3d(self, u, v, z_plane=None):
        """Pixel (u, v) -> 3D point in the robot frame, on the plane z = TRACKED_Z.
        Returns None if the ray is (almost) horizontal or points behind the camera.
        """
        if self.cam_K is None or self.cam_T is None:
            return None
        if z_plane is None:
            z_plane = self.object_z
        pixel = np.array([[[u, v]]], dtype=float)
        x, y = cv2.undistortPoints(pixel, self.cam_K, self.cam_D).reshape(2)

        direction_cam = np.array([x, y, 1.0])
        axes = MUJOCO_TO_OPENCV if self.camera_axes == "mujoco" else np.eye(3)
        direction_world = self.cam_T[:3, :3] @ axes @ direction_cam
        origin_world = self.cam_T[:3, 3]

        if abs(direction_world[2]) < 1e-6:
            return None
        t = (z_plane - origin_world[2]) / direction_world[2]
        if t < 0:
            return None

        return origin_world + t * direction_world

    def _project_box(self, u, v):
        """Infer centre from a visible wall of a known, axis-aligned box.

        Use box_visible_wall=none for a full centre silhouette. The default
        y-wall assumption is specific to the occluded box in scene_tek5.xml.
        """
        point = self._project_to_3d(u, v, z_plane=self.box_z)
        if point is not None and self.box_visible_wall != "none":
            axis = 0 if self.box_visible_wall == "x" else 1
            point[axis] -= np.sign(self.cam_T[axis, 3] - point[axis]) * self.box_half_width
        return point

    def _publish_result(self):
        """Publish the current detections in metres in the world frame.

        Call after processing an image. Clear positions to None when detection
        fails, so an old detection is not published as a new observation.
        """
        stamp = self._image_stamp or self.get_clock().now().to_msg()
        for position, publisher in (
            (self.object_position, self._pub_object),
            (self.drop_box_position, self._pub_drop_box),
        ):
            if position is None:
                continue
            point = np.asarray(position, dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                self.get_logger().warning("Ignoring invalid detected position")
                continue
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = self.output_frame
            msg.pose.position.x = float(point[0])
            msg.pose.position.y = float(point[1])
            msg.pose.position.z = float(point[2])
            msg.pose.orientation.w = 1.0
            publisher.publish(msg)

    def _fetch_cam_params(self):
        """Request flattened cam_K (9) and MuJoCo cam_T (16) from the driver.

        Both parameters must be ROS double arrays in row-major order. cam_T
        maps MuJoCo camera coordinates to world; _project_to_3d handles the
        OpenCV axis conversion. This retrieves calibration, not estimates it.
        """
        if self.cam_K is not None and self.cam_T is not None:
            self._cam_params_timer.cancel()
            return
        if self._cam_params_future is not None:
            return
        if not self._cam_params_client.service_is_ready():
            return

        request = GetParameters.Request()
        request.names = ["cam_K", "cam_T"]
        self._cam_params_future = self._cam_params_client.call_async(request)
        self._cam_params_future.add_done_callback(self._on_cam_params)

    def _on_cam_params(self, future):
        """Validate calibration before exposing it to the image callback."""
        self._cam_params_future = None
        try:
            response = future.result()
            if response is None or len(response.values) != 2:
                raise ValueError("expected cam_K and cam_T")
            if any(value.type != ParameterType.PARAMETER_DOUBLE_ARRAY
                   for value in response.values):
                raise ValueError("cam_K and cam_T must be double arrays")
            K = np.asarray(response.values[0].double_array_value).reshape(3, 3)
            T = np.asarray(response.values[1].double_array_value).reshape(4, 4)
            if not np.all(np.isfinite(K)) or not np.all(np.isfinite(T)):
                raise ValueError("camera matrices contain non-finite values")
            if K[0, 0] <= 0 or K[1, 1] <= 0:
                raise ValueError("camera focal lengths must be positive")
            if not np.allclose(K[2], [0, 0, 1]):
                raise ValueError("invalid intrinsic matrix")
            if not np.allclose(T[3], [0, 0, 0, 1]):
                raise ValueError("invalid homogeneous transform")
            R = T[:3, :3]
            if (not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(R), 1.0, atol=1e-5)):
                raise ValueError("camera transform must contain a proper rotation")
        except Exception as exc:
            self.get_logger().warning(f"Camera calibration unavailable: {exc}")
            return

        # CameraInfo takes precedence if its callback has already supplied K.
        if self.cam_K is None:
            self.cam_K = K
        self.cam_T = T
        self._cam_params_timer.cancel()
        self.get_logger().info("Camera calibration received from driver")

    @staticmethod
    def _validate_K(K):
        if (K.shape != (3, 3) or not np.all(np.isfinite(K))
                or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.allclose(K[2], [0, 0, 1])):
            raise ValueError("invalid camera intrinsic matrix")

    def _load_calibration(self, filename):
        """Load measured K, D and camera->world T from a JSON file."""
        data = json.loads(Path(filename).read_text())
        K = np.asarray(data["K"], dtype=float).reshape(3, 3)
        T = np.asarray(data["T"], dtype=float).reshape(4, 4)
        D = np.asarray(data.get("D", []), dtype=float)
        self._validate_K(K)
        if (not np.all(np.isfinite(T)) or not np.allclose(T[3], [0, 0, 0, 1])
                or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-5)):
            raise ValueError("invalid camera transform")
        if D.size not in (0, 4, 5, 8, 12, 14) or not np.all(np.isfinite(D)):
            raise ValueError("invalid distortion coefficients")
        axes = data.get("camera_axes", "opencv")
        if axes not in ("opencv", "mujoco"):
            raise ValueError("invalid camera axes")
        self.cam_K, self.cam_T, self.cam_D = K, T, D
        self.camera_axes = axes
        self.output_frame = data.get("frame_id", self.output_frame)
        if "image_size" in data:
            self._camera_size = tuple(data["image_size"])


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
