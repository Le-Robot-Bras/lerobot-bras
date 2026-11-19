#!/usr/bin/env python3
"""Detect OpenCV DICT_4X4_50 ArUco marker ID 0 on the desk as a world anchor."""
import json
import math
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _rotmat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 proper rotation matrix to (x, y, z, w) quaternion."""
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _orthogonalize(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation 3x3 matrix back onto SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0.0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    return R_ortho


class ArucoNode(Node):
    """Detect DICT_4X4_50 ArUco marker ID 0 to anchor the camera in base_link."""

    def __init__(self):
        super().__init__("aruco_node")
        self._bridge = CvBridge()

        self._frame_id = str(self.declare_parameter("frame_id", "base_link").value)
        self._marker_id = int(self.declare_parameter("marker_id", 0).value)
        self._marker_size_m = float(self.declare_parameter("marker_size_m", 0.05).value)
        self._marker_x = float(self.declare_parameter("marker_x", 0.51).value)
        self._marker_y = float(self.declare_parameter("marker_y", 0.0).value)
        self._marker_z = float(self.declare_parameter("marker_z", 0.0).value)
        self._marker_yaw_deg = float(self.declare_parameter("marker_yaw_deg", 0.0).value)
        self._camera_fov_deg = float(self.declare_parameter("camera_fov_deg", 60.0).value)
        self._smoothing_alpha = _clamp(
            float(self.declare_parameter("smoothing_alpha", 0.25).value), 0.01, 1.0
        )
        self._debug_terminal = bool(self.declare_parameter("debug_terminal", True).value)
        debug_rate_hz = float(self.declare_parameter("debug_rate_hz", 1.0).value)
        self._debug_period = 1.0 / max(debug_rate_hz, 0.1)
        self._last_debug_log = 0.0
        self._last_detected_state: bool | None = None

        if self._marker_size_m <= 0.0:
            raise ValueError("marker_size_m must be positive")

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

        s = self._marker_size_m / 2.0
        self._obj_points = np.array(
            [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
            dtype=np.float64,
        )

        yaw = math.radians(self._marker_yaw_deg)
        c, sn = math.cos(yaw), math.sin(yaw)
        self._T_world_marker = np.array(
            [
                [c, -sn, 0.0, self._marker_x],
                [sn, c, 0.0, self._marker_y],
                [0.0, 0.0, 1.0, self._marker_z],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self._cam_K: np.ndarray | None = None
        self._cam_D = np.zeros(5, dtype=np.float64)

        # Latched anchor state (survives hand occlusion once locked)
        self._locked = False
        self._T_world_cam: np.ndarray | None = None
        self._t_cam_marker: np.ndarray | None = None
        self._marker_center_px: list[float] | None = None
        self._marker_side_px: float | None = None
        self._active_K: np.ndarray | None = None

        self.create_subscription(CameraInfo, "/external_cam/camera_info", self._cb_camera_info, 10)
        self.create_subscription(Image, "/external_cam/image_raw", self._cb_image, 10)

        self._pub_anchor = self.create_publisher(String, "/aruco/anchor_state", 10)
        self._pub_pose = self.create_publisher(PoseStamped, "/aruco/pose", 10)
        self._pub_annotated = self.create_publisher(Image, "/aruco/image_annotated", 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            f"ArUco anchor node ready (DICT_4X4_50 id={self._marker_id}, "
            f"size={self._marker_size_m:.3f} m at "
            f"({self._marker_x:.2f}, {self._marker_y:.2f}, {self._marker_z:.2f}) in {self._frame_id})"
        )

    def _cb_camera_info(self, msg: CameraInfo):
        K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(K)) or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
            return
        self._cam_K = K
        D = np.asarray(msg.d, dtype=np.float64)
        self._cam_D = D if (D.size > 0 and np.all(np.isfinite(D))) else np.zeros(5, dtype=np.float64)

    def _get_intrinsics(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        if self._cam_K is not None:
            return self._cam_K, self._cam_D
        fov_rad = math.radians(_clamp(self._camera_fov_deg, 10.0, 170.0))
        fx = (width / 2.0) / math.tan(fov_rad / 2.0)
        K = np.array(
            [
                [fx, 0.0, width / 2.0],
                [0.0, fx, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return K, np.zeros(5, dtype=np.float64)

    def _solve_marker_pnp(
        self, corners_2d: np.ndarray, K: np.ndarray, D: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Estimate (rvec, tvec, R_cm) using IPPE_SQUARE and select the physical desk solution."""
        num, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            self._obj_points, corners_2d, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if num == 0 or not rvecs:
            return None

        best = None
        fallback = None
        for rvec, tvec, err in zip(rvecs, tvecs, errs):
            tvec_flat = np.asarray(tvec, dtype=np.float64).reshape(3)
            if tvec_flat[2] <= 0.02:
                continue
            R_cm, _ = cv2.Rodrigues(rvec)
            cam_in_marker = -R_cm.T @ tvec_flat
            err_val = float(np.asarray(err, dtype=np.float64).item())
            cand = (err_val, np.asarray(rvec, dtype=np.float64), tvec_flat, R_cm)
            if cam_in_marker[2] > 0.01:
                if best is None or err_val < best[0]:
                    best = cand
            elif fallback is None or err_val < fallback[0]:
                fallback = cand

        chosen = best if best is not None else fallback
        if chosen is None:
            return None
        return chosen[1], chosen[2], chosen[3]

    def _update_anchor(
        self, R_cm: np.ndarray, t_cm: np.ndarray, corners_2d: np.ndarray, K: np.ndarray
    ):
        """Compute T_world_cam and EMA-smooth & latch the anchor state."""
        R_mc = R_cm.T
        t_mc = -R_mc @ t_cm
        T_marker_cam = np.eye(4, dtype=np.float64)
        T_marker_cam[:3, :3] = R_mc
        T_marker_cam[:3, 3] = t_mc
        T_wc_raw = self._T_world_marker @ T_marker_cam

        edges = [
            float(np.linalg.norm(corners_2d[(k + 1) % 4] - corners_2d[k]))
            for k in range(4)
        ]
        side_px = sum(edges) / 4.0
        center_px = corners_2d.mean(axis=0)

        if not self._locked:
            self._T_world_cam = T_wc_raw
            self._t_cam_marker = t_cm.copy()
            self._marker_center_px = [float(center_px[0]), float(center_px[1])]
            self._marker_side_px = side_px
            self._locked = True
        else:
            a = self._smoothing_alpha
            assert self._T_world_cam is not None and self._t_cam_marker is not None
            assert self._marker_center_px is not None and self._marker_side_px is not None
            R_blend = (1.0 - a) * self._T_world_cam[:3, :3] + a * T_wc_raw[:3, :3]
            self._T_world_cam[:3, :3] = _orthogonalize(R_blend)
            self._T_world_cam[:3, 3] = (1.0 - a) * self._T_world_cam[:3, 3] + a * T_wc_raw[:3, 3]
            self._t_cam_marker = (1.0 - a) * self._t_cam_marker + a * t_cm
            self._marker_center_px = [
                (1.0 - a) * self._marker_center_px[0] + a * float(center_px[0]),
                (1.0 - a) * self._marker_center_px[1] + a * float(center_px[1]),
            ]
            self._marker_side_px = (1.0 - a) * self._marker_side_px + a * side_px

        self._active_K = K.copy()

    def _publish_tf_and_pose(self, stamp):
        transforms = []

        tf_marker = TransformStamped()
        tf_marker.header.stamp = stamp
        tf_marker.header.frame_id = self._frame_id
        tf_marker.child_frame_id = f"aruco_marker_{self._marker_id}"
        tf_marker.transform.translation.x = self._marker_x
        tf_marker.transform.translation.y = self._marker_y
        tf_marker.transform.translation.z = self._marker_z
        qx, qy, qz, qw = _rotmat_to_quat(self._T_world_marker[:3, :3])
        tf_marker.transform.rotation.x = qx
        tf_marker.transform.rotation.y = qy
        tf_marker.transform.rotation.z = qz
        tf_marker.transform.rotation.w = qw
        transforms.append(tf_marker)

        if self._locked and self._T_world_cam is not None:
            t_wc = self._T_world_cam[:3, 3]
            cx, cy, cz, cw = _rotmat_to_quat(self._T_world_cam[:3, :3])

            tf_cam = TransformStamped()
            tf_cam.header.stamp = stamp
            tf_cam.header.frame_id = self._frame_id
            tf_cam.child_frame_id = "real_camera"
            tf_cam.transform.translation.x = float(t_wc[0])
            tf_cam.transform.translation.y = float(t_wc[1])
            tf_cam.transform.translation.z = float(t_wc[2])
            tf_cam.transform.rotation.x = cx
            tf_cam.transform.rotation.y = cy
            tf_cam.transform.rotation.z = cz
            tf_cam.transform.rotation.w = cw
            transforms.append(tf_cam)

            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self._frame_id
            pose.pose.position.x = float(t_wc[0])
            pose.pose.position.y = float(t_wc[1])
            pose.pose.position.z = float(t_wc[2])
            pose.pose.orientation.x = cx
            pose.pose.orientation.y = cy
            pose.pose.orientation.z = cz
            pose.pose.orientation.w = cw
            self._pub_pose.publish(pose)

        self._tf_broadcaster.sendTransform(transforms)

    def _publish_anchor_state(self, detected: bool, K: np.ndarray):
        cam_dist = (
            float(np.linalg.norm(self._t_cam_marker))
            if self._t_cam_marker is not None
            else None
        )
        payload = {
            "detected": detected,
            "locked": self._locked,
            "marker_id": self._marker_id,
            "marker_size_m": self._marker_size_m,
            "marker_world_pos": [self._marker_x, self._marker_y, self._marker_z],
            "marker_yaw_deg": self._marker_yaw_deg,
            "t_cam_marker": (
                [round(float(v), 5) for v in self._t_cam_marker]
                if self._t_cam_marker is not None
                else None
            ),
            "T_world_cam": (
                self._T_world_cam.round(6).tolist()
                if self._T_world_cam is not None
                else None
            ),
            "cam_K": (self._active_K if self._active_K is not None else K).round(3).tolist(),
            "marker_center_px": (
                [round(float(v), 2) for v in self._marker_center_px]
                if self._marker_center_px is not None
                else None
            ),
            "marker_side_px": (
                round(float(self._marker_side_px), 2)
                if self._marker_side_px is not None
                else None
            ),
            "cam_distance_m": round(cam_dist, 4) if cam_dist is not None else None,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self._pub_anchor.publish(msg)

    def _log_debug(self, detected: bool):
        if not self._debug_terminal:
            return
        now = time.monotonic()
        state_changed = detected != self._last_detected_state
        if not state_changed and (now - self._last_debug_log) < self._debug_period:
            return
        self._last_detected_state = detected
        self._last_debug_log = now

        if detected and self._T_world_cam is not None and self._t_cam_marker is not None:
            cam_pos = self._T_world_cam[:3, 3]
            dist = float(np.linalg.norm(self._t_cam_marker))
            self.get_logger().info(
                f"[ARUCO ID:{self._marker_id}] LIVE | dist={dist:.3f}m | "
                f"side={self._marker_side_px:.1f}px | "
                f"cam_pos=({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})m"
            )
        elif self._locked:
            self.get_logger().info(
                f"[ARUCO ID:{self._marker_id}] LOCKED (occluded, holding calibration)"
            )
        else:
            self.get_logger().info(
                f"[ARUCO ID:{self._marker_id}] Searching for DICT_4X4_50 marker..."
            )

    def _cb_image(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warning(f"Image conversion failed: {exc}")
            return

        h, w = frame.shape[:2]
        K, D = self._get_intrinsics(w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self._detector.detectMarkers(gray)

        detected = False
        annotated = frame.copy()
        rvec_draw = None
        tvec_draw = None
        target_idx = None

        if ids is not None:
            for i, marker_id_arr in enumerate(ids):
                if int(marker_id_arr[0]) == self._marker_id:
                    target_idx = i
                    break

        if target_idx is not None:
            corners_2d = corners_list[target_idx].reshape(4, 2).astype(np.float64)
            pnp = self._solve_marker_pnp(corners_2d, K, D)
            if pnp is not None:
                rvec_draw, tvec_draw, R_cm = pnp
                self._update_anchor(R_cm, tvec_draw, corners_2d, K)
                detected = True

        if detected and target_idx is not None and rvec_draw is not None and tvec_draw is not None:
            cv2.aruco.drawDetectedMarkers(
                annotated,
                [corners_list[target_idx]],
                np.array([[self._marker_id]], dtype=np.int32),
            )
            cv2.drawFrameAxes(
                annotated, K, D, rvec_draw, tvec_draw, self._marker_size_m * 0.7
            )
            dist = float(np.linalg.norm(self._t_cam_marker)) if self._t_cam_marker is not None else 0.0
            banner = f"[ARUCO ID:{self._marker_id} LIVE | dist={dist:.2f}m]"
            banner_color = (0, 255, 0)
        elif self._locked and self._marker_center_px is not None:
            cx_i, cy_i = int(self._marker_center_px[0]), int(self._marker_center_px[1])
            cv2.drawMarker(
                annotated, (cx_i, cy_i), (0, 220, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA
            )
            banner = f"[ARUCO ID:{self._marker_id} LOCKED (occluded)]"
            banner_color = (0, 220, 255)
        else:
            banner = f"[ARUCO ID:{self._marker_id} SEARCHING...]"
            banner_color = (0, 140, 255)

        cv2.putText(
            annotated,
            banner,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            banner_color,
            2,
            cv2.LINE_AA,
        )

        stamp = msg.header.stamp if msg.header.stamp.sec or msg.header.stamp.nanosec else self.get_clock().now().to_msg()
        self._publish_tf_and_pose(stamp)
        self._publish_anchor_state(detected, K)
        self._log_debug(detected)

        out_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        self._pub_annotated.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
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
