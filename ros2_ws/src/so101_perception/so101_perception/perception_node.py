#!/usr/bin/env python3
"""perception_node.py - red cube and drop box, 3D in the robot frame.

Subscribes:  /external_cam/image_raw, /external_cam/camera_info
Publishes:   /object_position, /drop_box_position  (geometry_msgs/PoseStamped, 10 Hz max,
             position = object centre, orientation = yaw around z)
             /perception/mask  (sensor_msgs/Image mono8, for RViz)
             /perception/debug (sensor_msgs/Image bgr8: image + fitted models)
             /perception/markers (visualization_msgs/MarkerArray: 3D cube and box for RViz)

Method (see vision.py): HSV mask, then fit of a box of known size lying on the
table so that its projection matches the silhouette -> position + yaw, robust
to perspective and to the image border. A detection whose fit score is low
(object hidden by the arm...) is not published: the brain keeps its last value.

Camera pose, by priority:
  1. `calibration_file` (JSON from calibrate_camera: K, D, T camera->robot, OpenCV axes);
     this is what the real robot uses.
  2. otherwise cam_K / cam_T parameters of the driver (sim: ground-truth pose of the
     MuJoCo camera, MuJoCo axes).
K from /external_cam/camera_info overrides the driver one when no calibration file.
"""
import json
import time
from pathlib import Path

import numpy as np
import rclpy
import rclpy.node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray

from so101_perception import vision


def _valid_transform(T):
    R = T[:3, :3]
    return (np.all(np.isfinite(T)) and np.allclose(T[3], [0, 0, 0, 1])
            and np.allclose(R.T @ R, np.eye(3), atol=1e-5) and np.isclose(np.linalg.det(R), 1.0, atol=1e-5))


def _valid_K(K):
    return (K.shape == (3, 3) and np.all(np.isfinite(K)) and K[0, 0] > 0 and K[1, 1] > 0
            and np.allclose(K[2], [0, 0, 1]))


class PerceptionNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("so101_perception")
        self.bridge = CvBridge()
        publish_hz = float(self.declare_parameter("publish_hz", 10.0).value)
        if not np.isfinite(publish_hz) or publish_hz <= 0:
            raise ValueError("publish_hz must be positive")
        self._period = 1.0 / publish_hz
        self._min_score = float(self.declare_parameter("min_score", vision.MIN_SCORE).value)
        self._frame = self.declare_parameter("output_frame", "base_link").value

        self._K = None          # intrinsics
        self._D = None
        self._T = None          # camera -> robot
        self._axes = "mujoco"
        self._size = None
        self._camera = None     # vision.Camera, rebuilt when K/T change
        self._from_file = False
        self._last_publish = 0.0

        calibration_file = self.declare_parameter("calibration_file", "").value
        if calibration_file:
            self._load_calibration(calibration_file)

        self._params_future = None
        self._params_client = self.create_client(GetParameters, "/so101_driver/get_parameters")
        self._params_timer = None if self._from_file else self.create_timer(1.0, self._fetch_driver_camera)

        self._pub_object = self.create_publisher(PoseStamped, "/object_position", 10)
        self._pub_box = self.create_publisher(PoseStamped, "/drop_box_position", 10)
        self._pub_mask = self.create_publisher(Image, "/perception/mask", 2)
        self._pub_debug = self.create_publisher(Image, "/perception/debug", 2)
        self._pub_markers = self.create_publisher(MarkerArray, "/perception/markers", 2)
        self.create_subscription(CameraInfo, "/external_cam/camera_info", self._cb_camera_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, "/external_cam/image_raw", self._cb_image, qos_profile_sensor_data)
        self.get_logger().info("Perception ready; waiting for camera calibration and images")

    # ---------------------------------------------------------- calibration

    def _load_calibration(self, filename):
        data = json.loads(Path(filename).read_text())
        K = np.asarray(data["K"], dtype=float).reshape(3, 3)
        T = np.asarray(data["T"], dtype=float).reshape(4, 4)
        D = np.asarray(data.get("D", []), dtype=float)
        axes = data.get("camera_axes", "opencv")
        if not _valid_K(K) or not _valid_transform(T) or axes not in ("opencv", "mujoco"):
            raise ValueError(f"invalid calibration file {filename}")
        self._K, self._T, self._D, self._axes = K, T, D, axes
        self._frame = data.get("frame_id", self._frame)
        self._size = tuple(data["image_size"]) if "image_size" in data else None
        self._from_file = True
        self._rebuild_camera()
        self.get_logger().info(f"Camera calibration loaded from {filename}")

    def _cb_camera_info(self, msg: CameraInfo):
        if self._from_file:
            return
        K = np.asarray(msg.k, dtype=float).reshape(3, 3)
        if not _valid_K(K) or msg.width <= 0 or msg.height <= 0:
            self.get_logger().warning("Ignoring invalid CameraInfo", throttle_duration_sec=10.0)
            return
        if self._K is None or not np.allclose(K, self._K) or self._size != (msg.width, msg.height):
            self._K, self._D, self._size = K, np.asarray(msg.d, dtype=float), (msg.width, msg.height)
            self._rebuild_camera()

    def _fetch_driver_camera(self):
        """Get cam_T (and cam_K) from the driver parameters (simulation)."""
        if self._T is not None:
            self._params_timer.cancel()
            return
        if self._params_future is not None or not self._params_client.service_is_ready():
            return
        request = GetParameters.Request()
        request.names = ["cam_K", "cam_T"]
        self._params_future = self._params_client.call_async(request)
        self._params_future.add_done_callback(self._on_driver_camera)

    def _on_driver_camera(self, future):
        self._params_future = None
        try:
            values = future.result().values
            if len(values) != 2 or any(v.type != ParameterType.PARAMETER_DOUBLE_ARRAY for v in values):
                raise ValueError("driver has no cam_K / cam_T (real robot? use calibration_file)")
            K = np.asarray(values[0].double_array_value).reshape(3, 3)
            T = np.asarray(values[1].double_array_value).reshape(4, 4)
            if not _valid_K(K) or not _valid_transform(T):
                raise ValueError("invalid cam_K / cam_T")
        except Exception as exc:  # noqa: BLE001 - keep retrying, log once in a while
            self.get_logger().warning(f"Camera pose unavailable: {exc}", throttle_duration_sec=10.0)
            return
        if self._K is None:
            self._K = K
        self._T, self._axes = T, "mujoco"
        self._rebuild_camera()
        self.get_logger().info("Camera pose received from the driver")

    def _rebuild_camera(self):
        if self._K is not None and self._T is not None:
            self._camera = vision.Camera(self._K, self._T, self._D, axes=self._axes, size=self._size)

    # ------------------------------------------------------------ images

    def _cb_image(self, msg: Image):
        now = time.monotonic()
        if self._camera is None or now - self._last_publish < self._period:
            return
        if self._size and (msg.width, msg.height) != tuple(self._size):
            self.get_logger().warning("Image size differs from the calibration", throttle_duration_sec=10.0)
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warning(f"cv_bridge: {exc}", throttle_duration_sec=10.0)
            return
        self._last_publish = now

        cube, cube_mask = vision.detect_cube(image, self._camera)
        box, box_mask = vision.detect_drop_box(image, self._camera)
        markers = MarkerArray()
        for det, pub, kind in ((cube, self._pub_object, "cube"), (box, self._pub_box, "box")):
            if det is not None and det.score >= self._min_score and np.all(np.isfinite(det.position)):
                pose = self._pose(det, msg.header.stamp)
                pub.publish(pose)
                markers.markers.extend(self._markers(kind, pose))
        if markers.markers:
            self._pub_markers.publish(markers)

        if self._pub_mask.get_subscription_count():
            mask_msg = self.bridge.cv2_to_imgmsg(cube_mask | box_mask, encoding="mono8")
            mask_msg.header = msg.header
            self._pub_mask.publish(mask_msg)
        if self._pub_debug.get_subscription_count():
            debug_msg = self.bridge.cv2_to_imgmsg(self._debug_image(image, cube, box), encoding="bgr8")
            debug_msg.header = msg.header
            self._pub_debug.publish(debug_msg)

    def _pose(self, det: vision.Detection, stamp) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in det.position)
        msg.pose.orientation.z = float(np.sin(det.yaw / 2))
        msg.pose.orientation.w = float(np.cos(det.yaw / 2))
        return msg

    def _markers(self, kind, pose: PoseStamped):
        """RViz markers at the detected pose: a red 5 cm cube, or the open brown box
        (bottom + 4 walls). The perceived cube is semi-transparent and lives 1 s (raw
        detection; the brain draws the cube it believes in, e.g. in the gripper); the
        box stays at its last detection."""
        if kind == "cube":
            w, d, h = vision.CUBE_SIZE
            parts = [((0.0, 0.0, 0.0), (w, d, h))]
            color, lifetime = (1.0, 0.4, 0.5, 0.35), 1
        else:
            w, d, h = vision.BOX_SIZE
            t = 0.004
            z0 = -h / 2  # pose is the box centre
            parts = [((0.0, 0.0, z0 + t / 2), (w, d, t)),
                     ((0.0, d / 2 - t / 2, 0.0), (w, t, h)), ((0.0, -d / 2 + t / 2, 0.0), (w, t, h)),
                     ((w / 2 - t / 2, 0.0, 0.0), (t, d, h)), ((-w / 2 + t / 2, 0.0, 0.0), (t, d, h))]
            color, lifetime = (0.45, 0.27, 0.15, 1.0), 0
        q = pose.pose.orientation
        yaw = 2 * np.arctan2(q.z, q.w)
        c, s_ = np.cos(yaw), np.sin(yaw)
        out = []
        for i, ((ox, oy, oz), scale) in enumerate(parts):
            m = Marker()
            m.header = pose.header
            m.ns, m.id = ("perception_cube" if kind == "cube" else "box"), i
            m.type, m.action = Marker.CUBE, Marker.ADD
            m.pose.position.x = pose.pose.position.x + c * ox - s_ * oy
            m.pose.position.y = pose.pose.position.y + s_ * ox + c * oy
            m.pose.position.z = pose.pose.position.z + oz
            m.pose.orientation = q
            m.scale.x, m.scale.y, m.scale.z = scale
            m.color.r, m.color.g, m.color.b, m.color.a = color
            m.lifetime.sec = lifetime
            out.append(m)
        return out

    def _debug_image(self, image, cube, box):
        import cv2
        out = image.copy()
        for det, size, color in ((cube, vision.CUBE_SIZE, (0, 255, 0)), (box, vision.BOX_SIZE, (255, 200, 0))):
            if det is None:
                continue
            pts = self._camera.project(vision.box_corners(det.position[0], det.position[1], det.yaw, size))
            hull = cv2.convexHull(pts.astype(np.int32))
            ok = det.score >= self._min_score
            cv2.polylines(out, [hull], True, color if ok else (0, 0, 255), 1)
            cv2.putText(out, f"{det.score:.2f}", tuple(int(v) for v in pts.min(axis=0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color if ok else (0, 0, 255), 1)
        return out


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
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
