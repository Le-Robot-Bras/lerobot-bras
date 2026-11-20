#!/usr/bin/env python3
"""Project detected 2D webcam hands into 3D robot space (base_link) for RViz."""
import json
import math
import time

from geometry_msgs.msg import Point, PoseStamped, TransformStamped
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

# Vivid RGB colors in [0, 1] per detected gesture for the 3D hand ball
GESTURE_COLORS_RGB = {
    "OPEN_HAND":     (0.10, 0.95, 0.25),
    "CLOSED_HAND":   (0.95, 0.15, 0.15),
    "V_SIGN":        (0.00, 0.86, 1.00),
    "THUMBS_UP":     (1.00, 0.92, 0.10),
    "THUMBS_DOWN":   (1.00, 0.50, 0.05),
    "OK_SIGN":       (1.00, 0.15, 0.95),
    "PINCH":         (0.15, 0.55, 1.00),
    "POINTING":      (0.10, 1.00, 0.95),
    "ROCK":          (0.95, 0.35, 0.78),
    "THREE_FINGERS": (0.39, 1.00, 0.65),
    "FOUR_FINGERS":  (0.65, 1.00, 0.30),
    "PARTIAL":       (0.25, 0.70, 1.00),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _make_color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r = float(r)
    c.g = float(g)
    c.b = float(b)
    c.a = float(a)
    return c


def _make_point(x: float, y: float, z: float) -> Point:
    p = Point()
    p.x = float(x)
    p.y = float(y)
    p.z = float(z)
    return p


def _transform_point(T: list, pt: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = pt
    return (
        T[0][0] * x + T[0][1] * y + T[0][2] * z + T[0][3],
        T[1][0] * x + T[1][1] * y + T[1][2] * z + T[1][3],
        T[2][0] * x + T[2][1] * y + T[2][2] * z + T[2][3],
    )


class Hand3DVizNode(Node):
    """Map /hand/fingers_state into 3D base_link coordinates, TF, and RViz markers."""

    def __init__(self):
        super().__init__("hand_3d_node")

        self._frame_id = str(self.declare_parameter("frame_id", "base_link").value)
        self._x_min = float(self.declare_parameter("x_min", 0.12).value)
        self._x_max = float(self.declare_parameter("x_max", 0.32).value)
        self._default_x = float(self.declare_parameter("default_x", 0.22).value)
        self._y_min = float(self.declare_parameter("y_min", -0.22).value)
        self._y_max = float(self.declare_parameter("y_max", 0.22).value)
        self._z_min = float(self.declare_parameter("z_min", 0.03).value)
        self._z_max = float(self.declare_parameter("z_max", 0.32).value)
        self._mirror_y = bool(self.declare_parameter("mirror_y", True).value)
        self._use_depth_estimation = bool(
            self.declare_parameter("use_depth_estimation", True).value
        )
        self._palm_scale_min = float(self.declare_parameter("palm_scale_min", 0.06).value)
        self._palm_scale_max = float(self.declare_parameter("palm_scale_max", 0.20).value)
        self._smoothing_alpha = _clamp(
            float(self.declare_parameter("smoothing_alpha", 0.35).value), 0.01, 1.0
        )
        self._hand_ball_diameter = float(
            self.declare_parameter("hand_ball_diameter", 0.055).value
        )

        # ArUco world-anchor & wrist-tag parameters
        self._palm_to_marker_ratio = float(
            self.declare_parameter("palm_to_marker_ratio", 1.2).value
        )
        self._palm_size_m = float(self.declare_parameter("palm_size_m", 0.083).value)
        self._depth_smoothing_alpha = _clamp(
            float(self.declare_parameter("depth_smoothing_alpha", 0.25).value), 0.01, 1.0
        )
        self._aruco_gain = float(self.declare_parameter("aruco_gain", 1.0).value)
        self._use_full_6dof_rotation = bool(
            self.declare_parameter("use_full_6dof_rotation", True).value
        )
        self._clamp_aruco_to_workspace = bool(
            self.declare_parameter("clamp_aruco_to_workspace", False).value
        )
        self._floor_z_m = float(self.declare_parameter("floor_z_m", 0.0).value)
        self._invert_depth_x = bool(self.declare_parameter("invert_depth_x", False).value)
        self._use_wrist_aruco = bool(self.declare_parameter("use_wrist_aruco", True).value)
        self._wrist_aruco_timeout_s = float(
            self.declare_parameter("wrist_aruco_timeout_s", 0.35).value
        )

        self._smoothed_pos: dict[str, tuple[float, float, float]] = {}
        self._smoothed_z_cam: dict[str, float] = {}
        self._last_seen: dict[str, float] = {}
        self._active_hand_marker_keys: set[tuple[str, int]] = set()
        self._single_marker_active = False
        self._aruco_anchor: dict | None = None
        self._last_wrist_anchor_time = 0.0
        self._last_aruco_status: tuple[bool, bool] | None = None
        self._calibrated_palm_size_m: float | None = None
        self._using_wrist_depth: bool | None = None

        self.create_subscription(String, "/hand/fingers_state", self._cb_fingers_state, 10)
        self.create_subscription(String, "/aruco/anchor_state", self._cb_anchor_state, 10)
        self._pub_pose = self.create_publisher(PoseStamped, "/hand/virtual_pose", 10)
        self._pub_markers = self.create_publisher(MarkerArray, "/hand/virtual_markers", 10)
        self._pub_marker = self.create_publisher(Marker, "/hand/virtual_marker", 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Keep the ArUco desk square visible in RViz even when no hand is in frame
        self.create_timer(1.0, self._publish_scene_markers)

        self.get_logger().info(
            f"3D hand visualizer ready (frame={self._frame_id}, "
            f"6dof={self._use_full_6dof_rotation}, "
            f"use_wrist_aruco={self._use_wrist_aruco}, "
            f"clamp_aruco={self._clamp_aruco_to_workspace})"
        )

    def _cb_anchor_state(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Ignoring invalid JSON on /aruco/anchor_state")
            return
        if not isinstance(data, dict):
            return
        self._aruco_anchor = data
        if data.get("wrist_detected") and data.get("t_cam_wrist"):
            self._last_wrist_anchor_time = time.monotonic()

        status = (bool(data.get("locked")), bool(data.get("detected")))
        if status != self._last_aruco_status:
            self._last_aruco_status = status
            self._publish_scene_markers()

    def _has_valid_aruco_anchor(self) -> bool:
        a = self._aruco_anchor
        if not a or not a.get("locked"):
            return False
        K = a.get("cam_K")
        t_cm = a.get("t_cam_marker")
        m_world = a.get("marker_world_pos")
        side_px = a.get("marker_side_px")
        if (
            not K
            or len(K) != 3
            or not t_cm
            or len(t_cm) != 3
            or not m_world
            or len(m_world) != 3
            or not side_px
            or float(side_px) <= 0.0
        ):
            return False
        return float(K[0][0]) > 0.0 and float(K[1][1]) > 0.0 and float(t_cm[2]) > 0.0

    def _wrist_depth_for_hand(
        self, wrist_px: list, num_hands: int
    ) -> float | None:
        """Return ArUco ID 1 metric camera depth if freshly detected near this hand."""
        if not self._use_wrist_aruco or not self._aruco_anchor:
            return None
        a = self._aruco_anchor
        if not a.get("wrist_detected"):
            return None
        if (time.monotonic() - self._last_wrist_anchor_time) > self._wrist_aruco_timeout_s:
            return None
        t_cw = a.get("t_cam_wrist")
        if not t_cw or len(t_cw) != 3 or float(t_cw[2]) <= 0.05:
            return None

        if num_hands > 1:
            w_center = a.get("wrist_center_px")
            w_side = float(a.get("wrist_side_px") or 40.0)
            if w_center and len(w_center) == 2:
                dist_px = math.hypot(
                    float(wrist_px[0]) - float(w_center[0]),
                    float(wrist_px[1]) - float(w_center[1]),
                )
                if dist_px > max(3.0 * w_side, 140.0):
                    return None

        return float(t_cw[2])

    def _project_palm_to_3d(
        self,
        hand_key: str,
        palm_norm: list,
        palm_px: list,
        wrist_px: list,
        palm_span_px: float | None = None,
        wrist_z_cam: float | None = None,
    ) -> tuple[float, float, float]:
        """Project webcam palm coordinates to clamped & smoothed (x, y, z)."""
        u_px, v_px = float(palm_px[0]), float(palm_px[1])
        d_px = math.hypot(u_px - float(wrist_px[0]), v_px - float(wrist_px[1]))

        x_lo, x_hi = min(self._x_min, self._x_max), max(self._x_min, self._x_max)
        y_lo, y_hi = min(self._y_min, self._y_max), max(self._y_min, self._y_max)
        z_lo, z_hi = min(self._z_min, self._z_max), max(self._z_min, self._z_max)

        if self._has_valid_aruco_anchor():
            a = self._aruco_anchor
            assert a is not None
            K = a["cam_K"]
            fx, cx = float(K[0][0]), float(K[0][2])
            fy, cy = float(K[1][1]), float(K[1][2])
            f_avg = 0.5 * (fx + fy)
            tx_m, ty_m, tz_m = (float(v) for v in a["t_cam_marker"])
            mx, my, mz = (float(v) for v in a["marker_world_pos"])
            side_px = float(a["marker_side_px"])

            using_wrist = wrist_z_cam is not None
            if using_wrist != self._using_wrist_depth:
                self._using_wrist_depth = using_wrist
                if using_wrist:
                    self.get_logger().info(
                        f"Depth source switched to ArUco wrist ID 1 (z={wrist_z_cam:.3f}m)."
                    )
                else:
                    self.get_logger().info(
                        "Wrist ID 1 occluded; using calibrated MediaPipe palm depth."
                    )

            if wrist_z_cam is not None:
                raw_z_cam = _clamp(wrist_z_cam, 0.10, 2.0)
                if palm_span_px is not None and float(palm_span_px) > 15.0:
                    est_palm_m = _clamp(raw_z_cam * float(palm_span_px) / f_avg, 0.045, 0.14)
                    if self._calibrated_palm_size_m is None:
                        self._calibrated_palm_size_m = est_palm_m
                    else:
                        self._calibrated_palm_size_m = (
                            0.85 * self._calibrated_palm_size_m + 0.15 * est_palm_m
                        )
            elif self._use_depth_estimation:
                if palm_span_px is not None and float(palm_span_px) > 1.0:
                    eff_palm_m = self._calibrated_palm_size_m or self._palm_size_m
                    raw_z_cam = _clamp(
                        f_avg * eff_palm_m / max(float(palm_span_px), 15.0),
                        0.12,
                        2.0,
                    )
                else:
                    d_px_ref = max(self._palm_to_marker_ratio * side_px, 15.0)
                    raw_z_cam = _clamp(tz_m * (d_px_ref / max(d_px, 15.0)), 0.12, 2.0)
            else:
                raw_z_cam = tz_m

            prev_z = self._smoothed_z_cam.get(hand_key)
            if prev_z is None:
                z_cam = raw_z_cam
            elif wrist_z_cam is not None:
                # Wrist ArUco z is already smoothed in aruco_node; track it snappily
                z_cam = 0.65 * raw_z_cam + 0.35 * prev_z
            else:
                da = self._depth_smoothing_alpha
                z_cam = da * raw_z_cam + (1.0 - da) * prev_z
            self._smoothed_z_cam[hand_key] = z_cam

            x_cam = (u_px - cx) / fx * z_cam
            y_cam = (v_px - cy) / fy * z_cam

            p_cam = (
                tx_m + (x_cam - tx_m) * self._aruco_gain,
                ty_m + (y_cam - ty_m) * self._aruco_gain,
                tz_m + (z_cam - tz_m) * self._aruco_gain,
            )

            T_wc = a.get("T_world_cam")
            if self._use_full_6dof_rotation and T_wc and len(T_wc) == 4:
                raw_x, raw_y, raw_z = _transform_point(T_wc, p_cam)
            else:
                dx_cam, dy_cam, dz_cam = p_cam[0] - tx_m, p_cam[1] - ty_m, p_cam[2] - tz_m
                raw_x = (mx + dz_cam) if self._invert_depth_x else (mx - dz_cam)
                raw_y = my + (-dx_cam if self._mirror_y else dx_cam)
                raw_z = mz + (-dy_cam)

            if self._clamp_aruco_to_workspace:
                x_lo, x_hi = min(x_lo, mx), max(x_hi, mx)
                y_lo, y_hi = min(y_lo, my), max(y_hi, my)
                z_lo, z_hi = min(z_lo, mz), max(z_hi, mz)
            else:
                x_lo, x_hi = -2.0, 2.0
                y_lo, y_hi = -2.0, 2.0
                z_lo, z_hi = self._floor_z_m, 2.0
        else:
            nx = float(palm_norm[0])
            ny = float(palm_norm[1])

            raw_y = (
                self._y_max - nx * (self._y_max - self._y_min)
                if self._mirror_y
                else self._y_min + nx * (self._y_max - self._y_min)
            )
            raw_z = self._z_max - ny * (self._z_max - self._z_min)

            if self._use_depth_estimation:
                img_h = (
                    (v_px / ny)
                    if (ny > 0.05 and v_px > 0.0)
                    else 480.0
                )
                scale = d_px / max(img_h, 1.0)
                denom = max(self._palm_scale_max - self._palm_scale_min, 1e-6)
                t = _clamp((scale - self._palm_scale_min) / denom, 0.0, 1.0)
                raw_x = self._x_min + t * (self._x_max - self._x_min)
            else:
                raw_x = self._default_x

        raw_x = _clamp(raw_x, x_lo, x_hi)
        raw_y = _clamp(raw_y, y_lo, y_hi)
        raw_z = _clamp(raw_z, z_lo, z_hi)

        prev = self._smoothed_pos.get(hand_key)
        if prev is None:
            smoothed = (raw_x, raw_y, raw_z)
        else:
            a = self._smoothing_alpha
            smoothed = (
                a * raw_x + (1.0 - a) * prev[0],
                a * raw_y + (1.0 - a) * prev[1],
                a * raw_z + (1.0 - a) * prev[2],
            )
        self._smoothed_pos[hand_key] = smoothed
        self._last_seen[hand_key] = time.monotonic()
        return smoothed

    def _base_marker(self, stamp, ns: str, marker_id: int, marker_type: int) -> Marker:
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self._frame_id
        m.ns = ns
        m.id = marker_id
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def _build_aruco_square_markers(self, stamp) -> list[Marker]:
        """Build RViz square markers representing the physical ArUco marker on the desk."""
        a = self._aruco_anchor or {}
        size = float(a.get("marker_size_m", 0.10))
        mx, my, mz = (float(v) for v in a.get("marker_world_pos", [0.51, 0.0, 0.0]))
        yaw = math.radians(float(a.get("marker_yaw_deg", 90.0)))
        detected = bool(a.get("detected", False))
        locked = bool(a.get("locked", False))

        if detected:
            border_color = _make_color(0.10, 1.00, 0.25, 0.95)  # green = live
        elif locked:
            border_color = _make_color(0.00, 0.90, 1.00, 0.95)  # cyan = locked
        else:
            border_color = _make_color(0.85, 0.85, 0.85, 0.90)  # light grey = waiting

        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        # Flat white square base on the desk
        outer = self._base_marker(stamp, "aruco_square", 0, Marker.CUBE)
        outer.pose.position = _make_point(mx, my, mz + 0.0015)
        outer.pose.orientation.z = qz
        outer.pose.orientation.w = qw
        outer.scale.x = size
        outer.scale.y = size
        outer.scale.z = 0.003
        outer.color = _make_color(0.95, 0.95, 0.95, 0.95)

        # Inner dark square pattern
        inner = self._base_marker(stamp, "aruco_square", 1, Marker.CUBE)
        inner.pose.position = _make_point(mx, my, mz + 0.0020)
        inner.pose.orientation.z = qz
        inner.pose.orientation.w = qw
        inner.scale.x = size * 0.66
        inner.scale.y = size * 0.66
        inner.scale.z = 0.0035
        inner.color = _make_color(0.08, 0.08, 0.08, 0.98)

        # Colored square outline indicating detection state
        border = self._base_marker(stamp, "aruco_square", 2, Marker.LINE_LIST)
        border.pose.position = _make_point(mx, my, mz + 0.0035)
        border.pose.orientation.z = qz
        border.pose.orientation.w = qw
        border.scale.x = 0.0035
        border.color = border_color
        hs = size / 2.0
        quad = [
            _make_point(-hs, -hs, 0.0),
            _make_point(hs, -hs, 0.0),
            _make_point(hs, hs, 0.0),
            _make_point(-hs, hs, 0.0),
        ]
        for k in range(4):
            border.points.append(quad[k])
            border.points.append(quad[(k + 1) % 4])

        return [outer, inner, border]

    def _publish_scene_markers(self):
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        arr.markers.extend(self._build_aruco_square_markers(stamp))
        self._pub_markers.publish(arr)

    def _cb_fingers_state(self, msg: String):
        try:
            hands = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Ignoring invalid JSON on /hand/fingers_state")
            return

        if not isinstance(hands, list):
            return

        now = time.monotonic()
        stamp = self.get_clock().now().to_msg()
        marker_array = MarkerArray()
        marker_array.markers.extend(self._build_aruco_square_markers(stamp))

        new_marker_keys: set[tuple[str, int]] = set()
        seen_labels: dict[str, int] = {}
        tf_transforms: list[TransformStamped] = []

        for idx, hand in enumerate(hands):
            palm_norm = hand.get("palm_norm")
            palm_px = hand.get("palm_px_f") or hand.get("palm_px")
            wrist_px = hand.get("wrist_px")
            if (
                not palm_norm
                or len(palm_norm) < 2
                or not palm_px
                or len(palm_px) < 2
                or not wrist_px
                or len(wrist_px) < 2
            ):
                continue

            label = str(hand.get("hand", "Unknown"))
            occ = seen_labels.get(label, 0)
            seen_labels[label] = occ + 1
            hand_key = label if occ == 0 else f"{label}_{occ}"

            palm_span_px = hand.get("palm_span_px")
            wrist_z_cam = self._wrist_depth_for_hand(wrist_px, len(hands))
            px, py, pz = self._project_palm_to_3d(
                hand_key,
                palm_norm,
                palm_px,
                wrist_px,
                float(palm_span_px) if palm_span_px is not None else None,
                wrist_z_cam=wrist_z_cam,
            )
            gesture = str(hand.get("gesture", "PARTIAL"))
            r, g, b = GESTURE_COLORS_RGB.get(gesture, (0.25, 0.70, 1.00))

            # Colored ball for the hand
            ball = self._base_marker(stamp, "virtual_hand_ball", idx, Marker.SPHERE)
            ball.pose.position = _make_point(px, py, pz)
            ball.scale.x = self._hand_ball_diameter
            ball.scale.y = self._hand_ball_diameter
            ball.scale.z = self._hand_ball_diameter
            ball.color = _make_color(r, g, b, 0.95)
            marker_array.markers.append(ball)
            new_marker_keys.add(("virtual_hand_ball", idx))

            if idx == 0:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = stamp
                pose_msg.header.frame_id = self._frame_id
                pose_msg.pose.position.x = px
                pose_msg.pose.position.y = py
                pose_msg.pose.position.z = pz
                pose_msg.pose.orientation.w = 1.0
                self._pub_pose.publish(pose_msg)

                single = self._base_marker(stamp, "virtual_hand_single", 0, Marker.SPHERE)
                single.pose.position = _make_point(px, py, pz)
                single.scale.x = self._hand_ball_diameter
                single.scale.y = self._hand_ball_diameter
                single.scale.z = self._hand_ball_diameter
                single.color = _make_color(r, g, b, 0.95)
                self._pub_marker.publish(single)
                self._single_marker_active = True

            # Broadcast TF frame (virtual_hand, virtual_hand_2, ...)
            tf_msg = TransformStamped()
            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = self._frame_id
            tf_msg.child_frame_id = "virtual_hand" if idx == 0 else f"virtual_hand_{idx + 1}"
            tf_msg.transform.translation.x = px
            tf_msg.transform.translation.y = py
            tf_msg.transform.translation.z = pz
            tf_msg.transform.rotation.w = 1.0
            tf_transforms.append(tf_msg)

        # Remove stale hand balls when a hand disappears from frame
        for ns, mid in self._active_hand_marker_keys - new_marker_keys:
            del_m = Marker()
            del_m.header.stamp = stamp
            del_m.header.frame_id = self._frame_id
            del_m.ns = ns
            del_m.id = mid
            del_m.action = Marker.DELETE
            marker_array.markers.append(del_m)
        self._active_hand_marker_keys = new_marker_keys

        if not new_marker_keys and self._single_marker_active:
            del_single = Marker()
            del_single.header.stamp = stamp
            del_single.header.frame_id = self._frame_id
            del_single.ns = "virtual_hand_single"
            del_single.id = 0
            del_single.action = Marker.DELETE
            self._pub_marker.publish(del_single)
            self._single_marker_active = False

        # Expire smoothing state for hands absent longer than 0.5 s
        stale_keys = [k for k, ts in self._last_seen.items() if (now - ts) > 0.5]
        for k in stale_keys:
            self._last_seen.pop(k, None)
            self._smoothed_pos.pop(k, None)
            self._smoothed_z_cam.pop(k, None)

        if tf_transforms:
            self._tf_broadcaster.sendTransform(tf_transforms)
        self._pub_markers.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = Hand3DVizNode()
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
