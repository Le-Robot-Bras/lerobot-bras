#!/usr/bin/env python3
"""hand_control_node.py - executes arm movement patterns from hand states.

Gestures (from /hand/fingers_state + 3D palm on /hand/virtual_pose):
  - POINTING    : move arm straight up
  - OK_SIGN     : return to default/original position
  - OPEN_HAND   : slowly follow the palm to touch it; on touch, stop and look up;
                  stop if the hand goes out of range
  - CLOSED_HAND : avoid the hand while respecting safe joint, table, and base
                  limits; stop calmly if cornered so the arm is never forced
  - V_SIGN      : enter Rock-Paper-Scissors mode against the hand that made V_SIGN

Subscribes to:
  - /hand/fingers_state (std_msgs/String JSON from hand_node)
  - /hand/virtual_pose  (geometry_msgs/PoseStamped 3D palm in base_link from hand_3d_node)
  - /joint_states       (sensor_msgs/JointState from driver_node)

Publishes:
  - /joint_command      (sensor_msgs/JointState, arm joints in rad + optional gripper %)
"""
from enum import Enum, auto
import json
import math
from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
import numpy as np
import pinocchio
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from so101_brain.rps_game import RockPaperScissorsGame, RpsConfig, RpsPhase

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
IK_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
]
JOINT_NAMES = ARM_JOINT_NAMES + ["gripper"]

BASE_FRAME = "base_link"
TCP_FRAME = "gripper_frame_link"
ACCEPTED_TARGET_FRAMES = ("", BASE_FRAME, "world")

# Frames monitored during hand avoidance for table clearance, base keep-out, and distance.
FLOOR_CHECK_FRAMES = ("lower_arm_link", "wrist_link", "gripper_link", "gripper_frame_link")
BASE_CHECK_FRAMES = ("wrist_link", "gripper_link", "gripper_frame_link")
AVOID_CHECK_FRAMES = ("gripper_frame_link", "wrist_link")

# Approximate shoulder pivot and base keep-out cylinder in base_link (so101.urdf).
SHOULDER_PIVOT = np.array([0.0388, 0.0, 0.1166], dtype=float)
BASE_KEEPOUT_XY = np.array([0.015, 0.0], dtype=float)
BASE_KEEPOUT_RADIUS_M = 0.095
BASE_KEEPOUT_HEIGHT_M = 0.15

# Default resting/original pose (rad) if no initial /joint_states was observed yet.
ORIGINAL_POSE = {name: 0.0 for name in ARM_JOINT_NAMES}

# Joint angles (rad) placing the SO-ARM101 straight up:
# shoulder_lift=0 points the upper arm vertically; elbow_flex=-pi/2 aligns
# the forearm vertically above it with wrist_flex=0 keeping the gripper upright.
STRAIGHT_UP_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": -math.pi / 2.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
}

IK_MAX_ITERS = 150
IK_CONVERGENCE_TOL = 3e-3  # m
IK_DAMPING = 1e-2
IK_MAX_STEP = 0.2  # rad per IK iteration
IK_RESTARTS = 4


class ControlMode(Enum):
    """Active continuous control mode for interactive hand patterns."""

    IDLE = auto()
    OPEN_FOLLOWING = auto()
    OPEN_OUT_OF_RANGE = auto()
    OPEN_TOUCHED_LOOKING_UP = auto()
    CLOSED_AVOIDING = auto()
    CLOSED_SAFE = auto()
    CLOSED_CORNERED = auto()
    RPS_MODE = auto()


_OPEN_MODES = (
    ControlMode.OPEN_FOLLOWING,
    ControlMode.OPEN_OUT_OF_RANGE,
    ControlMode.OPEN_TOUCHED_LOOKING_UP,
)
_CLOSED_MODES = (
    ControlMode.CLOSED_AVOIDING,
    ControlMode.CLOSED_SAFE,
    ControlMode.CLOSED_CORNERED,
)


class HandControlNode(Node):
    """Maps detected hand gestures/movements to robot arm motion patterns."""

    def __init__(self):
        super().__init__("so101_hand_control")

        control_rate_hz = float(self.declare_parameter("control_rate_hz", 20.0).value)
        self._dt = 1.0 / max(control_rate_hz, 1.0)
        self._follow_speed_rad_s = float(self.declare_parameter("follow_speed_rad_s", 0.55).value)
        self._look_up_speed_rad_s = float(self.declare_parameter("look_up_speed_rad_s", 1.2).value)
        self._touch_distance_m = float(self.declare_parameter("touch_distance_m", 0.035).value)
        self._ik_tolerance_m = float(self.declare_parameter("ik_tolerance_m", 0.015).value)
        self._min_reach_m = float(self.declare_parameter("min_reach_m", 0.08).value)
        self._max_reach_m = float(self.declare_parameter("max_reach_m", 0.42).value)
        self._min_z_m = float(self.declare_parameter("min_z_m", 0.02).value)

        # Avoidance safety parameters for CLOSED_HAND.
        self._avoid_distance_m = float(self.declare_parameter("avoid_distance_m", 0.24).value)
        self._avoid_speed_rad_s = float(self.declare_parameter("avoid_speed_rad_s", 0.80).value)
        self._avoid_joint_margin_rad = float(
            self.declare_parameter("avoid_joint_margin_rad", 0.15).value
        )
        self._avoid_min_z_m = float(self.declare_parameter("avoid_min_z_m", 0.05).value)
        self._avoid_min_progress_m = float(
            self.declare_parameter("avoid_min_progress_m", 1.5e-3).value
        )

        self._pose_timeout_s = float(self.declare_parameter("pose_timeout_s", 0.5).value)
        self._lost_timeout_s = float(self.declare_parameter("lost_timeout_s", 0.5).value)
        self._rearm_timeout_s = float(self.declare_parameter("rearm_timeout_s", 0.8).value)
        self._max_cmd_lead_rad = float(self.declare_parameter("max_cmd_lead_rad", 0.15).value)

        # Rock-Paper-Scissors trigger & timing parameters.
        self._rps_v_sign_hold_s = float(self.declare_parameter("rps_v_sign_hold_s", 0.35).value)
        rps_beat_duration_s = float(self.declare_parameter("rps_beat_duration_s", 1.20).value)
        rps_stance_hold_s = float(self.declare_parameter("rps_stance_hold_s", 1.5).value)
        self._rps_cooldown_s = float(self.declare_parameter("rps_cooldown_s", 1.0).value)

        urdf = Path(get_package_share_directory("so101_description")) / "urdf" / "so101.urdf"
        self._model = pinocchio.buildModelFromUrdf(str(urdf))
        self._data = self._model.createData()
        self._tcp_frame_id = self._model.getFrameId(TCP_FRAME)
        self._floor_frame_ids = [self._model.getFrameId(n) for n in FLOOR_CHECK_FRAMES]
        self._base_frame_ids = [self._model.getFrameId(n) for n in BASE_CHECK_FRAMES]
        self._avoid_frame_ids = [self._model.getFrameId(n) for n in AVOID_CHECK_FRAMES]

        self._idx_q = {n: self._model.joints[self._model.getJointId(n)].idx_q for n in JOINT_NAMES}
        self._idx_v = {n: self._model.joints[self._model.getJointId(n)].idx_v for n in JOINT_NAMES}
        self._arm_idx_q = [self._idx_q[n] for n in ARM_JOINT_NAMES]
        self._ik_idx_q = [self._idx_q[n] for n in IK_JOINT_NAMES]
        self._ik_idx_v = [self._idx_v[n] for n in IK_JOINT_NAMES]
        self._rng = np.random.default_rng()

        # Soft joint limits pulled inward from URDF hard stops for safe avoidance.
        mid = 0.5 * (self._model.lowerPositionLimit + self._model.upperPositionLimit)
        self._safe_lo = np.minimum(
            self._model.lowerPositionLimit + self._avoid_joint_margin_rad, mid
        )
        self._safe_hi = np.maximum(
            self._model.upperPositionLimit - self._avoid_joint_margin_rad, mid
        )

        self._current_joints: dict[str, float] = {}
        self._q_measured: np.ndarray | None = None
        self._original_pose: dict[str, float] = dict(ORIGINAL_POSE)
        self._original_pose_captured: bool = False
        self._active_pattern: str | None = None

        # Latest 3D palm pose published by hand_3d_node (/hand/virtual_pose, primary hand idx=0).
        self._hand_pose: np.ndarray | None = None
        self._hand_pose_time: float = 0.0

        # Shared interactive control state.
        self._mode = ControlMode.IDLE
        self._last_mode_gesture_time: float = 0.0
        self._mode_start_time: float = 0.0
        self._cmd_q: np.ndarray | None = None
        self._lookup_target_q: np.ndarray | None = None
        self._last_ik_q: np.ndarray | None = None
        self._ik_cache: tuple[np.ndarray, np.ndarray | None] | None = None
        self._avoid_pan_pref: float = 0.0

        # Rock-Paper-Scissors game instance and trigger tracking.
        self._rps = RockPaperScissorsGame(
            config=RpsConfig(
                beat_duration_s=rps_beat_duration_s,
                stance_hold_s=rps_stance_hold_s,
            ),
            solve_height_pose=self._solve_rps_height_pose,
            log=self.get_logger().info,
        )
        self._v_sign_since: float | None = None
        self._v_sign_hand_label: str | None = None
        self._rps_cooldown_until: float = 0.0
        self._rps_straight_seed: tuple[float, float] = (0.0, 0.0)
        self._rps_lookup_seed: tuple[float, float] = (1.2, -1.2)

        # Dispatch table from detected gesture name to pattern handler.
        self._gesture_patterns = {
            "POINTING": self._pattern_straight_up,
            "OK_SIGN": self._pattern_original_position,
            "OPEN_HAND": self._pattern_open_hand,
            "CLOSED_HAND": self._pattern_closed_hand,
        }

        self._pub_command = self.create_publisher(JointState, "/joint_command", 10)
        self.create_subscription(JointState, "/joint_states", self._cb_joint_states, 10)
        self.create_subscription(String, "/hand/fingers_state", self._cb_hand_state, 10)
        self.create_subscription(PoseStamped, "/hand/virtual_pose", self._cb_hand_pose, 10)
        self.create_timer(self._dt, self._control_step)

        self.get_logger().info(
            "Hand control node ready (POINTING=up, OK_SIGN=default, "
            "OPEN_HAND=touch+lookup, CLOSED_HAND=avoid, V_SIGN=rps)."
        )

    def _cb_joint_states(self, msg: JointState) -> None:
        """Store measured joint positions (rad) and capture the initial pose."""
        q = self._q_measured.copy() if self._q_measured is not None else pinocchio.neutral(self._model)
        for name, pos in zip(msg.name, msg.position):
            val = float(pos)
            self._current_joints[name] = val
            if name in self._idx_q:
                q[self._idx_q[name]] = val
        self._q_measured = q

        if not self._original_pose_captured and all(
            name in self._current_joints for name in ARM_JOINT_NAMES
        ):
            self._original_pose = {name: self._current_joints[name] for name in ARM_JOINT_NAMES}
            self._original_pose_captured = True

    def _cb_hand_pose(self, msg: PoseStamped) -> None:
        """Cache the latest 3D palm position in base_link from hand_3d_node."""
        frame = msg.header.frame_id
        if frame not in ACCEPTED_TARGET_FRAMES:
            self.get_logger().warning(
                f"Ignoring /hand/virtual_pose in unsupported frame '{frame}' (expected '{BASE_FRAME}')",
                throttle_duration_sec=5.0,
            )
            return

        pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=float,
        )
        if not np.all(np.isfinite(pos)):
            return

        self._hand_pose = pos
        self._hand_pose_time = time.monotonic()

    def _cb_hand_state(self, msg: String) -> None:
        """Dispatch the first recognized gesture in /hand/fingers_state."""
        try:
            hands = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning(
                "Received invalid JSON on /hand/fingers_state",
                throttle_duration_sec=5.0,
            )
            return

        if not isinstance(hands, list):
            return

        now = time.monotonic()
        pose_fresh = (
            self._hand_pose is not None
            and (now - self._hand_pose_time) <= self._pose_timeout_s
        )

        if self._mode == ControlMode.RPS_MODE:
            for hand_data in hands:
                if isinstance(hand_data, dict) and hand_data.get("gesture") == "OK_SIGN":
                    self._rps.reset()
                    self._reset_interactive_mode()
                    self._active_pattern = "OK_SIGN"
                    self.get_logger().info("RPS aborted by OK_SIGN; returning to default pose.")
                    self._pattern_original_position(hand_data)
                    return
            self._rps.update_hands(hands, self._hand_pose, pose_fresh, now)
            return

        v_sign_match = next(
            (
                (idx, hd)
                for idx, hd in enumerate(hands)
                if isinstance(hd, dict) and hd.get("gesture") == "V_SIGN"
            ),
            None,
        )
        if v_sign_match is not None and now >= self._rps_cooldown_until:
            v_idx, v_hand = v_sign_match
            v_label = str(v_hand.get("hand", "Unknown"))
            if self._v_sign_since is None or self._v_sign_hand_label != v_label:
                self._v_sign_since = now
                self._v_sign_hand_label = v_label
            elif (now - self._v_sign_since) >= self._rps_v_sign_hold_s:
                self._reset_interactive_mode()
                self._mode = ControlMode.RPS_MODE
                self._active_pattern = "V_SIGN"
                self._mode_start_time = now
                self._cmd_q = self._q_measured.copy() if self._q_measured is not None else None
                self._v_sign_since = None
                self._v_sign_hand_label = None
                self._rps_straight_seed = (0.0, 0.0)
                self._rps_lookup_seed = (1.2, -1.2)
                initial_z = (
                    float(self._hand_pose[2])
                    if (v_idx == 0 and pose_fresh and self._hand_pose is not None)
                    else None
                )
                self._rps.start(v_label, now, initial_z=initial_z)
                self._rps.update_hands(hands, self._hand_pose, pose_fresh, now)
                return
        else:
            self._v_sign_since = None
            self._v_sign_hand_label = None

        for idx, hand_data in enumerate(hands):
            if not isinstance(hand_data, dict):
                continue
            gesture = hand_data.get("gesture")
            handler = self._gesture_patterns.get(gesture)
            if handler is None:
                continue

            if gesture in ("OPEN_HAND", "CLOSED_HAND"):
                handler(hand_data, idx)
                return

            # Any explicit discrete gesture cancels interactive tracking/avoidance immediately.
            self._reset_interactive_mode()
            if self._active_pattern != gesture:
                hand_label = hand_data.get("hand", "Unknown")
                self.get_logger().info(
                    f"Executing pattern for gesture '{gesture}' ({hand_label} hand)."
                )
                self._active_pattern = gesture
            handler(hand_data)
            return

        if self._mode == ControlMode.IDLE:
            self._active_pattern = None

    def _pattern_straight_up(self, _hand_data: dict) -> None:
        """Command the arm straight up when the POINTING gesture is seen."""
        self._send_joint_command(STRAIGHT_UP_POSE)

    def _pattern_original_position(self, _hand_data: dict) -> None:
        """Return the arm to its default/captured original pose on OK_SIGN."""
        self._send_joint_command(self._original_pose)

    def _pattern_open_hand(self, hand_data: dict, hand_idx: int) -> None:
        """Start or keep alive the slow palm-touch pattern on OPEN_HAND."""
        if hand_idx != 0:
            self.get_logger().warning(
                "OPEN_HAND detected on secondary hand; /hand/virtual_pose only tracks the primary hand.",
                throttle_duration_sec=5.0,
            )
            return

        now = time.monotonic()
        self._last_mode_gesture_time = now

        if self._mode not in _OPEN_MODES:
            hand_label = hand_data.get("hand", "Unknown")
            self._reset_interactive_mode()
            self._mode = ControlMode.OPEN_FOLLOWING
            self._active_pattern = "OPEN_HAND"
            self._mode_start_time = now
            self._cmd_q = self._q_measured.copy() if self._q_measured is not None else None
            self.get_logger().info(
                f"Executing pattern for gesture 'OPEN_HAND' ({hand_label} hand): "
                "slowly moving to touch palm."
            )

    def _pattern_closed_hand(self, hand_data: dict, hand_idx: int) -> None:
        """Start or keep alive the safe hand-avoidance pattern on CLOSED_HAND."""
        if hand_idx != 0:
            self.get_logger().warning(
                "CLOSED_HAND detected on secondary hand; /hand/virtual_pose only tracks the primary hand.",
                throttle_duration_sec=5.0,
            )
            return

        now = time.monotonic()
        self._last_mode_gesture_time = now

        if self._mode not in _CLOSED_MODES:
            hand_label = hand_data.get("hand", "Unknown")
            self._reset_interactive_mode()
            self._mode = ControlMode.CLOSED_AVOIDING
            self._active_pattern = "CLOSED_HAND"
            self._mode_start_time = now
            self._cmd_q = self._q_measured.copy() if self._q_measured is not None else None
            self.get_logger().info(
                f"Executing pattern for gesture 'CLOSED_HAND' ({hand_label} hand): "
                "avoiding hand within safe limits."
            )

    def _reset_interactive_mode(self) -> None:
        """Clear interactive state for OPEN_HAND and CLOSED_HAND patterns."""
        self._mode = ControlMode.IDLE
        self._cmd_q = None
        self._lookup_target_q = None
        self._last_ik_q = None
        self._ik_cache = None
        self._avoid_pan_pref = 0.0

    def _control_step(self) -> None:
        """Periodic 20 Hz loop driving OPEN_HAND, CLOSED_HAND, and RPS_MODE."""
        if self._mode == ControlMode.IDLE or self._q_measured is None:
            return

        now = time.monotonic()
        if self._mode == ControlMode.RPS_MODE:
            self._step_rps_mode(now)
        elif self._mode in _OPEN_MODES:
            self._step_open_hand_mode(now)
        elif self._mode in _CLOSED_MODES:
            self._step_closed_hand_mode(now)

    # ------------------------------------------------------------------ RPS_MODE

    def _step_rps_mode(self, now: float) -> None:
        """Advance Rock-Paper-Scissors mode and command arm + gripper smoothly."""
        assert self._q_measured is not None
        cmd = self._rps.step(now)
        if cmd is None or self._rps.phase == RpsPhase.FINISHED:
            self._rps_cooldown_until = now + self._rps_cooldown_s
            self._rps.reset()
            self._reset_interactive_mode()
            self._active_pattern = None
            return

        self._ensure_cmd_q_bounded()
        assert self._cmd_q is not None
        q_target = self._cmd_q.copy()
        for name, val in cmd.joints.items():
            if name in self._idx_q:
                q_target[self._idx_q[name]] = float(val)
        q_target = np.clip(
            q_target, self._model.lowerPositionLimit, self._model.upperPositionLimit
        )
        self._cmd_q = self._step_arm_joints(
            self._cmd_q, q_target, cmd.speed_rad_s * self._dt
        )
        self._send_joint_command(
            self._q_to_arm_dict(self._cmd_q), gripper_pct=cmd.gripper_pct
        )

    def _solve_rps_height_pose(self, target_z: float, look_up: bool) -> dict[str, float]:
        """Solve (shoulder_lift, elbow_flex, wrist_flex) to match target_z cleanly."""
        sl_idx = self._idx_q["shoulder_lift"]
        ef_idx = self._idx_q["elbow_flex"]
        wf_idx = self._idx_q["wrist_flex"]
        sl_lo = float(self._model.lowerPositionLimit[sl_idx]) + 0.05
        sl_hi = float(self._model.upperPositionLimit[sl_idx]) - 0.05
        ef_lo = float(self._model.lowerPositionLimit[ef_idx]) + 0.05
        ef_hi = float(self._model.upperPositionLimit[ef_idx]) - 0.05
        wf_lo = float(self._model.lowerPositionLimit[wf_idx])
        wf_hi = float(self._model.upperPositionLimit[wf_idx])

        pitch = -math.pi / 2.0 if look_up else 0.0
        target_x = 0.32 if look_up else 0.31
        # With the gripper pointing straight up (+0.16 m above wrist), z < 0.22 m would push the wrist into the table.
        target_z = float(np.clip(target_z, 0.22 if look_up else 0.10, 0.35))
        sl, ef = self._rps_lookup_seed if look_up else self._rps_straight_seed
        q_tmp = pinocchio.neutral(self._model)

        def tcp_xz(sl_val: float, ef_val: float) -> np.ndarray:
            q_tmp[sl_idx] = sl_val
            q_tmp[ef_idx] = ef_val
            q_tmp[wf_idx] = float(np.clip(pitch - (sl_val + ef_val), wf_lo, wf_hi))
            p = self._forward_kinematics(q_tmp).translation
            return np.array([float(p[0]), float(p[2])], dtype=float)

        eps = 1e-4
        for _ in range(50):
            p = tcp_xz(sl, ef)
            err = np.array([target_x - p[0], target_z - p[1]], dtype=float)
            if float(np.linalg.norm(err)) < 1e-3:
                break
            jac = np.column_stack(
                [
                    (tcp_xz(sl + eps, ef) - p) / eps,
                    (tcp_xz(sl, ef + eps) - p) / eps,
                ]
            )
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(2), err)
            dq = np.clip(dq, -0.15, 0.15)
            sl = float(np.clip(sl + dq[0], sl_lo, sl_hi))
            ef = float(np.clip(ef + dq[1], ef_lo, ef_hi))

        if look_up:
            self._rps_lookup_seed = (sl, ef)
        else:
            self._rps_straight_seed = (sl, ef)
        wf = float(np.clip(pitch - sl - ef, wf_lo, wf_hi))
        return {
            "shoulder_pan": 0.0,
            "shoulder_lift": sl,
            "elbow_flex": ef,
            "wrist_flex": wf,
            "wrist_roll": 0.0,
        }

    # ------------------------------------------------------------------ OPEN_HAND

    def _step_open_hand_mode(self, now: float) -> None:
        """Slow palm-following, touch detection + look-up, and out-of-range stop."""
        assert self._q_measured is not None
        gesture_age = now - self._last_mode_gesture_time

        if self._mode == ControlMode.OPEN_TOUCHED_LOOKING_UP:
            self._step_look_up()
            if gesture_age > self._rearm_timeout_s:
                self.get_logger().info("Open hand cleared; ready to touch palm again.")
                self._reset_interactive_mode()
                self._active_pattern = None
            return

        if gesture_age > self._lost_timeout_s:
            if self._mode == ControlMode.OPEN_FOLLOWING:
                self._hold_current_position()
                self.get_logger().info("Open hand lost / out of view; stopping.")
            self._reset_interactive_mode()
            self._active_pattern = None
            return

        pose_fresh = (
            self._hand_pose is not None
            and (now - self._hand_pose_time) <= self._pose_timeout_s
        )
        if not pose_fresh:
            if (now - self._mode_start_time) > self._pose_timeout_s:
                if self._mode != ControlMode.OPEN_OUT_OF_RANGE:
                    self._hold_current_position()
                    self._mode = ControlMode.OPEN_OUT_OF_RANGE
                    self.get_logger().info(
                        "3D palm pose (/hand/virtual_pose) unavailable; stopping."
                    )
            return

        assert self._hand_pose is not None
        palm_pos = self._hand_pose

        tcp_pos = self._forward_kinematics(self._q_measured).translation
        dist_to_palm = float(np.linalg.norm(tcp_pos - palm_pos))
        if dist_to_palm <= self._touch_distance_m:
            self._enter_touched_look_up(dist_to_palm)
            return

        q_target = self._solve_palm_target_ik(palm_pos)
        if q_target is None:
            if self._mode != ControlMode.OPEN_OUT_OF_RANGE:
                self._hold_current_position()
                self._mode = ControlMode.OPEN_OUT_OF_RANGE
                self.get_logger().info(
                    f"Palm target ({palm_pos[0]:.3f}, {palm_pos[1]:.3f}, {palm_pos[2]:.3f}) "
                    "is out of range; stopping."
                )
            return

        if self._mode == ControlMode.OPEN_OUT_OF_RANGE:
            self._mode = ControlMode.OPEN_FOLLOWING
            self._cmd_q = self._q_measured.copy()
            self.get_logger().info("Palm back in range; resuming slow follow.")

        self._ensure_cmd_q_bounded()
        assert self._cmd_q is not None
        self._cmd_q = self._step_arm_joints(
            self._cmd_q, q_target, self._follow_speed_rad_s * self._dt
        )
        self._send_joint_command(self._q_to_arm_dict(self._cmd_q))

    def _enter_touched_look_up(self, dist_to_palm: float) -> None:
        """Stop the arm where it touched the palm and tilt the wrist upward to look up."""
        assert self._q_measured is not None
        q_stop = self._q_measured.copy()
        q_lookup = q_stop.copy()

        # In so101.urdf, shoulder_lift + elbow_flex + wrist_flex == -pi/2 aligns the gripper straight up.
        sl = float(q_stop[self._idx_q["shoulder_lift"]])
        ef = float(q_stop[self._idx_q["elbow_flex"]])
        wf_idx = self._idx_q["wrist_flex"]
        wf_lo = float(self._model.lowerPositionLimit[wf_idx])
        wf_hi = float(self._model.upperPositionLimit[wf_idx])
        q_lookup[wf_idx] = float(np.clip(-math.pi / 2.0 - sl - ef, wf_lo, wf_hi))

        self._mode = ControlMode.OPEN_TOUCHED_LOOKING_UP
        self._cmd_q = q_stop
        self._lookup_target_q = q_lookup
        self._ik_cache = None

        self.get_logger().info(
            f"Touched palm (dist={dist_to_palm * 100.0:.1f} cm)! "
            f"Stopping and looking up (wrist_flex -> {q_lookup[wf_idx]:.2f} rad)."
        )
        self._step_look_up()

    def _step_look_up(self) -> None:
        """Smoothly complete the wrist tilt-up motion while keeping the rest of the arm stopped."""
        if self._lookup_target_q is None:
            return
        if self._cmd_q is None:
            assert self._q_measured is not None
            self._cmd_q = self._q_measured.copy()

        self._cmd_q = self._step_arm_joints(
            self._cmd_q, self._lookup_target_q, self._look_up_speed_rad_s * self._dt
        )
        self._send_joint_command(self._q_to_arm_dict(self._cmd_q))

        remaining = max(
            abs(float(self._cmd_q[idx] - self._lookup_target_q[idx]))
            for idx in self._arm_idx_q
        )
        if remaining < 1e-3:
            self._lookup_target_q = None

    # ------------------------------------------------------------------ CLOSED_HAND

    def _step_closed_hand_mode(self, now: float) -> None:
        """Avoid the closed hand safely; stop calmly if cornered at robot limits."""
        assert self._q_measured is not None
        gesture_age = now - self._last_mode_gesture_time

        if gesture_age > self._lost_timeout_s:
            if self._mode == ControlMode.CLOSED_AVOIDING:
                self._hold_current_position()
                self.get_logger().info("Closed hand lost / out of view; stopping avoidance.")
            self._reset_interactive_mode()
            self._active_pattern = None
            return

        pose_fresh = (
            self._hand_pose is not None
            and (now - self._hand_pose_time) <= self._pose_timeout_s
        )
        if not pose_fresh:
            if (now - self._mode_start_time) > self._pose_timeout_s:
                if self._mode == ControlMode.CLOSED_AVOIDING:
                    self._hold_current_position()
                    self._mode = ControlMode.CLOSED_SAFE
                    self.get_logger().info(
                        "3D hand pose (/hand/virtual_pose) unavailable; holding position."
                    )
            return

        assert self._hand_pose is not None
        hand_pos = self._hand_pose

        if self._mode != ControlMode.CLOSED_AVOIDING:
            self._cmd_q = self._q_measured.copy()
        self._ensure_cmd_q_bounded()
        assert self._cmd_q is not None

        q_next, status = self._compute_avoid_step(self._cmd_q, hand_pos)
        if status == "SAFE":
            if self._mode == ControlMode.CLOSED_AVOIDING:
                self._hold_current_position()
                self.get_logger().info("Safe distance from closed hand reached; holding.")
            self._mode = ControlMode.CLOSED_SAFE
            return

        if status == "CORNERED" or q_next is None:
            if self._mode != ControlMode.CLOSED_CORNERED:
                self._hold_current_position()
                self._mode = ControlMode.CLOSED_CORNERED
                self.get_logger().info(
                    "Cannot avoid closed hand further (reached safe robot limits); "
                    "holding position."
                )
            return

        if self._mode != ControlMode.CLOSED_AVOIDING:
            self._mode = ControlMode.CLOSED_AVOIDING
            self.get_logger().info("Evading closed hand.")

        self._cmd_q = q_next
        self._send_joint_command(self._q_to_arm_dict(self._cmd_q))

    def _update_all_frames(self, q: np.ndarray) -> None:
        pinocchio.forwardKinematics(self._model, self._data, q)
        pinocchio.updateFramePlacements(self._model, self._data)

    def _safety_metrics(self, q: np.ndarray) -> tuple[float, float]:
        """Return (min_link_z, max_base_penetration) across monitored arm frames."""
        self._update_all_frames(q)
        min_z = min(float(self._data.oMf[fid].translation[2]) for fid in self._floor_frame_ids)
        base_pen = 0.0
        for fid in self._base_frame_ids:
            p = self._data.oMf[fid].translation
            r = math.hypot(
                float(p[0]) - float(BASE_KEEPOUT_XY[0]),
                float(p[1]) - float(BASE_KEEPOUT_XY[1]),
            )
            if r < BASE_KEEPOUT_RADIUS_M and float(p[2]) < BASE_KEEPOUT_HEIGHT_M:
                pen = min(BASE_KEEPOUT_RADIUS_M - r, BASE_KEEPOUT_HEIGHT_M - float(p[2]))
                base_pen = max(base_pen, pen)
        return min_z, base_pen

    def _is_safe_avoid_transition(self, q_curr: np.ndarray, q_next: np.ndarray) -> bool:
        """Allow moves that keep all links above the desk and outside the base cylinder."""
        z_curr, pen_curr = self._safety_metrics(q_curr)
        z_next, pen_next = self._safety_metrics(q_next)
        if z_next < self._avoid_min_z_m and z_next < z_curr:
            return False
        if pen_next > 0.0 and pen_next > pen_curr:
            return False
        return True

    def _hand_clearance(self, q: np.ndarray, hand_pos: np.ndarray) -> float:
        """Minimum distance from monitored distal frames (TCP + wrist) to the hand."""
        self._update_all_frames(q)
        return min(
            float(np.linalg.norm(self._data.oMf[fid].translation - hand_pos))
            for fid in self._avoid_frame_ids
        )

    def _apply_safe_avoid_delta(
        self, q_curr: np.ndarray, dq_ik: np.ndarray, max_step: float
    ) -> np.ndarray | None:
        """Project a candidate joint velocity onto soft joint limits and step bounds."""
        # If a joint started slightly beyond the soft margin, allow moving inward only.
        eff_lo = np.maximum(np.minimum(q_curr, self._safe_lo), self._model.lowerPositionLimit)
        eff_hi = np.minimum(np.maximum(q_curr, self._safe_hi), self._model.upperPositionLimit)

        dq = np.asarray(dq_ik, dtype=float).copy()
        for k, qi in enumerate(self._ik_idx_q):
            if q_curr[qi] >= eff_hi[qi] - 1e-6 and dq[k] > 0.0:
                dq[k] = 0.0
            if q_curr[qi] <= eff_lo[qi] + 1e-6 and dq[k] < 0.0:
                dq[k] = 0.0

        max_abs = float(np.max(np.abs(dq)))
        if max_abs < 1e-9:
            return None

        dq *= min(1.0, max_step / max_abs)
        q_next = q_curr.copy()
        for k, qi in enumerate(self._ik_idx_q):
            q_next[qi] = float(np.clip(q_curr[qi] + dq[k], eff_lo[qi], eff_hi[qi]))
        return q_next

    def _compute_avoid_step(
        self, q_curr: np.ndarray, hand_pos: np.ndarray
    ) -> tuple[np.ndarray | None, str]:
        """Compute one safe joint step away from hand_pos, or return SAFE / CORNERED."""
        c_curr = self._hand_clearance(q_curr, hand_pos)
        if c_curr >= self._avoid_distance_m:
            return None, "SAFE"

        # Analytical distance-ascent gradient J^T u over TCP and wrist (singularity-safe).
        self._update_all_frames(q_curr)
        grad = np.zeros(len(self._ik_idx_v), dtype=float)
        w_sum = 0.0
        for fid in self._avoid_frame_ids:
            p_f = self._data.oMf[fid].translation
            v = p_f - hand_pos
            d = float(np.linalg.norm(v))
            u = v / d if d > 1e-4 else np.array([0.0, 0.0, 1.0])
            if float(p_f[2]) <= self._avoid_min_z_m + 0.03 and u[2] < 0.0:
                u = np.array([u[0], u[1], 0.0], dtype=float)
                u_norm = float(np.linalg.norm(u))
                u = u / u_norm if u_norm > 1e-4 else np.array([0.0, 0.0, 1.0])

            w = 1.0 / (max(d, 0.03) ** 2)
            jac = pinocchio.computeFrameJacobian(
                self._model, self._data, q_curr, fid, pinocchio.LOCAL_WORLD_ALIGNED
            )[:3, self._ik_idx_v]
            grad += w * (jac.T @ u)
            w_sum += w
        if w_sum > 0.0:
            grad /= w_sum

        pan_idx = self._idx_q["shoulder_pan"]
        if self._avoid_pan_pref == 0.0:
            pan_q = float(q_curr[pan_idx])
            self._avoid_pan_pref = (
                1.0
                if (float(self._safe_hi[pan_idx]) - pan_q) >= (pan_q - float(self._safe_lo[pan_idx]))
                else -1.0
            )

        # Evaluate steepest gradient plus boundary-sliding and pan/lift tie-breakers.
        candidates: list[np.ndarray] = [grad]
        for k in range(len(self._ik_idx_v)):
            if abs(float(grad[k])) > 1e-4:
                axis_vec = np.zeros(len(self._ik_idx_v), dtype=float)
                axis_vec[k] = math.copysign(1.0, float(grad[k]))
                candidates.append(axis_vec)
        candidates.append(np.array([self._avoid_pan_pref, 0.0, 0.0, 0.0], dtype=float))
        candidates.append(np.array([-self._avoid_pan_pref, 0.0, 0.0, 0.0], dtype=float))
        candidates.append(np.array([0.0, -1.0, -1.0, 0.0], dtype=float))

        max_step = self._avoid_speed_rad_s * self._dt
        best_q: np.ndarray | None = None
        best_score = 0.0
        for idx, cand in enumerate(candidates):
            q_cand = self._apply_safe_avoid_delta(q_curr, cand, max_step)
            if q_cand is None or not self._is_safe_avoid_transition(q_curr, q_cand):
                continue
            gain = self._hand_clearance(q_cand, hand_pos) - c_curr
            # Slightly prefer the smooth Jacobian gradient when gains are comparable.
            score = gain * (1.08 if idx == 0 else 1.0)
            if gain >= self._avoid_min_progress_m and score > best_score:
                best_score = score
                best_q = q_cand

        if best_q is None:
            return None, "CORNERED"

        pan_delta = float(best_q[pan_idx] - q_curr[pan_idx])
        if abs(pan_delta) > 1e-4:
            self._avoid_pan_pref = 1.0 if pan_delta > 0.0 else -1.0
        return best_q, "AVOIDING"

    # ------------------------------------------------------------------ Shared helpers

    def _ensure_cmd_q_bounded(self) -> None:
        """Initialize _cmd_q if needed and clamp lead relative to measured joints."""
        assert self._q_measured is not None
        if self._cmd_q is None:
            self._cmd_q = self._q_measured.copy()
            return
        for idx in self._arm_idx_q:
            self._cmd_q[idx] = float(
                np.clip(
                    self._cmd_q[idx],
                    self._q_measured[idx] - self._max_cmd_lead_rad,
                    self._q_measured[idx] + self._max_cmd_lead_rad,
                )
            )

    def _hold_current_position(self) -> None:
        """Publish the arm's current measured joint angles once to halt motion immediately."""
        if self._q_measured is None:
            return
        self._cmd_q = self._q_measured.copy()
        self._send_joint_command(self._q_to_arm_dict(self._q_measured))

    def _forward_kinematics(self, q: np.ndarray):
        """Return the TCP placement (pinocchio.SE3) in base_link."""
        self._update_all_frames(q)
        return self._data.oMf[self._tcp_frame_id]

    def _is_within_reach_bounds(self, target: np.ndarray) -> bool:
        """Fast geometric range guard before solving IK."""
        if target[2] < self._min_z_m:
            return False
        dist_from_pivot = float(np.linalg.norm(target - SHOULDER_PIVOT))
        return self._min_reach_m <= dist_from_pivot <= self._max_reach_m

    def _solve_palm_target_ik(self, target: np.ndarray) -> np.ndarray | None:
        """Solve position-only IK for the palm target, returning q or None if out of range."""
        if not self._is_within_reach_bounds(target):
            self._ik_cache = (target.copy(), None)
            return None

        if self._ik_cache is not None:
            cached_pos, cached_q = self._ik_cache
            if float(np.linalg.norm(target - cached_pos)) < 1e-3:
                return None if cached_q is None else cached_q.copy()

        lower, upper = self._model.lowerPositionLimit, self._model.upperPositionLimit
        q_base = (
            self._last_ik_q.copy()
            if self._last_ik_q is not None
            else (
                self._cmd_q.copy()
                if self._cmd_q is not None
                else (
                    self._q_measured.copy()
                    if self._q_measured is not None
                    else pinocchio.neutral(self._model)
                )
            )
        )
        keep_wrist_roll = float(q_base[self._idx_q["wrist_roll"]])
        keep_gripper = float(q_base[self._idx_q["gripper"]])

        # Seed order: continuous warm-start, neutral (escapes straight-up singularity), then random restarts.
        neutral = pinocchio.neutral(self._model)
        neutral[self._idx_q["wrist_roll"]] = keep_wrist_roll
        neutral[self._idx_q["gripper"]] = keep_gripper
        seeds = [q_base, neutral]
        for _ in range(IK_RESTARTS):
            seed = self._rng.uniform(lower, upper)
            seed[self._idx_q["wrist_roll"]] = keep_wrist_roll
            seed[self._idx_q["gripper"]] = keep_gripper
            seeds.append(seed)

        for seed in seeds:
            q_sol, err = self._solve_ik_from(target, seed)
            if err <= self._ik_tolerance_m:
                self._last_ik_q = q_sol.copy()
                self._ik_cache = (target.copy(), q_sol.copy())
                return q_sol

        self._ik_cache = (target.copy(), None)
        return None

    def _solve_ik_from(self, target: np.ndarray, q_seed: np.ndarray) -> tuple[np.ndarray, float]:
        """Damped least squares IK over the 4 positioning arm joints."""
        lower, upper = self._model.lowerPositionLimit, self._model.upperPositionLimit
        q = q_seed.copy()

        for _ in range(IK_MAX_ITERS):
            err = target - self._forward_kinematics(q).translation
            if float(np.linalg.norm(err)) < IK_CONVERGENCE_TOL:
                break

            jac = pinocchio.computeFrameJacobian(
                self._model, self._data, q, self._tcp_frame_id, pinocchio.LOCAL_WORLD_ALIGNED
            )[:3, self._ik_idx_v]
            dq_ik = jac.T @ np.linalg.solve(jac @ jac.T + IK_DAMPING**2 * np.eye(3), err)
            dq_ik = np.clip(dq_ik, -IK_MAX_STEP, IK_MAX_STEP)

            dq = np.zeros(self._model.nv)
            dq[self._ik_idx_v] = dq_ik
            q = np.clip(pinocchio.integrate(self._model, q, dq), lower, upper)

        final_err = float(np.linalg.norm(target - self._forward_kinematics(q).translation))
        return q, final_err

    def _step_arm_joints(
        self, q_current: np.ndarray, q_target: np.ndarray, max_step: float
    ) -> np.ndarray:
        """Advance arm joints proportionally toward q_target by at most max_step rad per joint."""
        q_next = q_current.copy()
        deltas = np.array([q_target[idx] - q_current[idx] for idx in self._arm_idx_q], dtype=float)
        max_delta = float(np.max(np.abs(deltas)))
        if max_delta <= 1e-9:
            return q_next

        scale = min(1.0, max_step / max_delta)
        for idx, delta in zip(self._arm_idx_q, deltas):
            q_next[idx] = q_current[idx] + scale * delta
        return np.clip(q_next, self._model.lowerPositionLimit, self._model.upperPositionLimit)

    def _q_to_arm_dict(self, q: np.ndarray) -> dict[str, float]:
        """Convert a Pinocchio configuration vector into an arm joint dictionary."""
        return {name: float(q[self._idx_q[name]]) for name in ARM_JOINT_NAMES}

    def _send_joint_command(
        self, target_joints: dict[str, float], gripper_pct: float | None = None
    ) -> None:
        """Publish an arm JointState command (rad) and optional gripper (0-100 %) to /joint_command."""
        self._original_pose_captured = True
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(ARM_JOINT_NAMES)
        positions = []
        for name in ARM_JOINT_NAMES:
            idx = self._idx_q[name]
            val = float(target_joints.get(name, self._current_joints.get(name, 0.0)))
            val = float(
                np.clip(val, self._model.lowerPositionLimit[idx], self._model.upperPositionLimit[idx])
            )
            positions.append(val)
        if gripper_pct is not None:
            msg.name.append("gripper")
            positions.append(float(np.clip(gripper_pct, 0.0, 100.0)))
        msg.position = positions
        self._pub_command.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HandControlNode()
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
