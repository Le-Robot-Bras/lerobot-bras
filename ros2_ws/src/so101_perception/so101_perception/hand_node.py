#!/usr/bin/env python3
"""MediaPipe Hand, Finger & Gesture Recognition Node (combined with YOLO view)."""
import json
import math
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path("/ros2_ws/hand_landmarker.task")

# Landmark indices for each finger: (MCP, PIP/IP, Tip)
FINGER_LANDMARKS = {
    "thumb":  (2, 3, 4),
    "index":  (5, 6, 8),
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20),
}

# Full bone chain from wrist (0) to fingertip for drawing independent finger lines
FINGER_CHAINS = {
    "thumb":  [0, 1, 2, 3, 4],
    "index":  [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring":   [0, 13, 14, 15, 16],
    "pinky":  [0, 17, 18, 19, 20],
}

PALM_BRIDGES = [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)]
PALM_LANDMARKS = (0, 5, 9, 13, 17)
# Rigid palm segments (wrist->MCPs + knuckle width) unaffected by finger curl
PALM_SPAN_EDGES = ((0, 5), (0, 9), (0, 13), (0, 17), (5, 17))

GESTURE_COLORS = {
    "OPEN_HAND":     (0, 255, 0),
    "CLOSED_HAND":   (0, 0, 255),
    "V_SIGN":        (255, 220, 0),
    "THUMBS_UP":     (0, 255, 255),
    "THUMBS_DOWN":   (0, 140, 255),
    "OK_SIGN":       (255, 0, 255),
    "PINCH":         (255, 128, 0),
    "POINTING":      (255, 255, 0),
    "ROCK":          (200, 100, 255),
    "THREE_FINGERS": (180, 255, 100),
    "FOUR_FINGERS":  (100, 255, 180),
    "PARTIAL":       (200, 200, 200),
}


def _dist(lm_a, lm_b) -> float:
    return math.hypot(lm_a.x - lm_b.x, lm_a.y - lm_b.y)


def _palm_size(lm) -> float:
    return max(_dist(lm[0], lm[9]), _dist(lm[0], lm[5]), _dist(lm[5], lm[17]), 1e-4)


class HandNode(Node):
    def __init__(self):
        super().__init__("hand_node")
        self.bridge = CvBridge()

        self.debug_terminal = bool(self.declare_parameter("debug_terminal", True).value)
        debug_rate_hz = float(self.declare_parameter("debug_rate_hz", 2.0).value)
        self._debug_period = 1.0 / max(debug_rate_hz, 0.1)
        self._last_debug_log_time = 0.0
        self._last_debug_signature = None

        if not MODEL_PATH.is_file():
            self.get_logger().info(f"Downloading MediaPipe model to {MODEL_PATH}...")
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.get_logger().info(
            "MediaPipe HandLandmarker initialized (gestures + terminal debug enabled)!"
        )

        self._latest_result = None

        # 1. Detect hands on the clean, un-occluded camera image (before YOLO draws boxes on top)
        self.sub_raw = self.create_subscription(
            Image, "/external_cam/image_raw", self.raw_image_callback, 10
        )
        # 2. Draw the detected finger lines on top of YOLO's bounding-box image
        self.sub_yolo = self.create_subscription(
            Image, "/yolo/image_boxes", self.yolo_image_callback, 10
        )

        # Publish combined YOLO + MediaPipe image on both topics so RViz displays both seamlessly
        self.pub_yolo_combined = self.create_publisher(Image, "/yolo/image_annotated", 10)
        self.pub_mp = self.create_publisher(Image, "/mediapipe/image_annotated", 10)

        # Structured finger + gesture + position state (for brain_node & hand_debug to consume)
        self.pub_state = self.create_publisher(String, "/hand/fingers_state", 10)

    def _get_finger_states(self, hand_landmarks) -> dict[str, int]:
        """Return 1 (extended) or 0 (curled) for each of the 5 fingers."""
        lm = hand_landmarks
        wrist = lm[0]
        pinky_mcp = lm[17]
        palm_sz = _palm_size(lm)

        states = {}

        # Thumb (cmc=1, mcp=2, ip=3, tip=4): check straightness & separation from palm/curled fingers
        thumb_straight = (
            _dist(lm[4], lm[2]) > _dist(lm[3], lm[2]) * 1.18
            and _dist(lm[4], lm[1]) > _dist(lm[3], lm[1]) * 1.15
        )
        thumb_tucked = (
            _dist(lm[4], lm[6]) < 0.35 * palm_sz
            or _dist(lm[4], lm[10]) < 0.38 * palm_sz
            or _dist(lm[4], lm[9]) < 0.38 * palm_sz
        )
        frontal_spread = (
            _dist(lm[4], pinky_mcp) > _dist(lm[3], pinky_mcp) * 1.05
            and _dist(lm[4], lm[5]) > 0.42 * palm_sz
            and _dist(lm[4], lm[6]) > 0.35 * palm_sz
        )
        sideways_extended = (
            _dist(lm[4], lm[5]) > 0.45 * palm_sz
            and _dist(lm[4], lm[6]) > 0.45 * palm_sz
            and _dist(lm[4], lm[10]) > 0.48 * palm_sz
        )
        states["thumb"] = int(
            thumb_straight and not thumb_tucked and (frontal_spread or sideways_extended)
        )

        # Other 4 fingers: require extension from both wrist (0) and the finger's MCP joint
        for name in ("index", "middle", "ring", "pinky"):
            mcp_idx, pip_idx, tip_idx = FINGER_LANDMARKS[name]
            wrist_ext = _dist(lm[tip_idx], wrist) > _dist(lm[pip_idx], wrist) * 1.10
            mcp_ext = _dist(lm[tip_idx], lm[mcp_idx]) > _dist(lm[pip_idx], lm[mcp_idx]) * 1.25
            states[name] = int(wrist_ext and mcp_ext)

        return states

    def _classify_gesture(self, hand_landmarks, finger_states: dict[str, int]) -> str:
        """Classify a hand landmark set and finger states into a named hand configuration."""
        lm = hand_landmarks
        palm_sz = _palm_size(lm)
        thumb_index_dist = _dist(lm[4], lm[8])
        is_ok_pinch = thumb_index_dist < 0.42 * palm_sz

        t = finger_states["thumb"]
        i = finger_states["index"]
        m = finger_states["middle"]
        r = finger_states["ring"]
        p = finger_states["pinky"]
        total = t + i + m + r + p

        # 1. OK sign: thumb & index tips touching, remaining 3 fingers extended
        if is_ok_pinch and m == 1 and r == 1 and p == 1:
            return "OK_SIGN"

        # 2 & 3. Thumbs Up / Thumbs Down: 4 fingers curled, thumb extended vertically
        if i == 0 and m == 0 and r == 0 and p == 0 and (t == 1 or _dist(lm[4], lm[5]) > 0.42 * palm_sz):
            if lm[4].y < lm[3].y and (lm[2].y - lm[4].y) > 0.18 * palm_sz and lm[4].y < lm[5].y:
                return "THUMBS_UP"
            if lm[4].y > lm[3].y and (lm[4].y - lm[2].y) > 0.18 * palm_sz and lm[4].y > lm[5].y:
                return "THUMBS_DOWN"

        # 4. Fully Open Hand: all 5 fingers extended and thumb/index not pinching
        if total == 5 and not is_ok_pinch:
            return "OPEN_HAND"

        # 5. Fully Closed Hand (fist): all 5 fingers curled
        if t == 0 and i == 0 and m == 0 and r == 0 and p == 0:
            return "CLOSED_HAND"

        # 6. V Sign (2 fingers): index & middle extended, ring & pinky curled, thumb tucked/curled
        if (
            i == 1
            and m == 1
            and r == 0
            and p == 0
            and not is_ok_pinch
            and (t == 0 or _dist(lm[4], lm[13]) < 0.65 * palm_sz or _dist(lm[4], lm[14]) < 0.65 * palm_sz)
        ):
            return "V_SIGN"

        # 7. Pinch: thumb & index pinching while middle/ring/pinky curled
        if is_ok_pinch and m == 0 and r == 0 and p == 0:
            return "PINCH"

        # 8. Pointing: index only
        if i == 1 and m == 0 and r == 0 and p == 0 and t == 0:
            return "POINTING"

        # 9. Rock / Horns: index & pinky extended, middle & ring curled
        if i == 1 and m == 0 and r == 0 and p == 1:
            return "ROCK"

        # Fallbacks for other combinations
        if total == 3:
            return "THREE_FINGERS"
        if total == 4:
            return "FOUR_FINGERS"
        return "PARTIAL"

    def _draw_hands(self, annotated, result):
        """Draw skeletons, gesture banners, palm positions, and build structured payload."""
        h, w, _ = annotated.shape
        hands_payload = []
        if result is None or not result.hand_landmarks:
            return hands_payload

        for hand_lms, handedness_list in zip(result.hand_landmarks, result.handedness):
            label = handedness_list[0].category_name if handedness_list else "Unknown"
            finger_states = self._get_finger_states(hand_lms)
            gesture = self._classify_gesture(hand_lms, finger_states)

            # Convert all 21 normalized landmarks to pixel coordinates
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

            # Palm center, tilt-compensated 3D palm span (px), & bounding box
            avg_x = sum(hand_lms[idx].x for idx in PALM_LANDMARKS) / len(PALM_LANDMARKS)
            avg_y = sum(hand_lms[idx].y for idx in PALM_LANDMARKS) / len(PALM_LANDMARKS)
            avg_z = sum(hand_lms[idx].z for idx in PALM_LANDMARKS) / len(PALM_LANDMARKS)
            palm_px = [int(avg_x * w), int(avg_y * h)]
            palm_px_f = [round(avg_x * w, 2), round(avg_y * h, 2)]
            palm_span_px = sum(
                math.sqrt(
                    ((hand_lms[a].x - hand_lms[b].x) * w) ** 2
                    + ((hand_lms[a].y - hand_lms[b].y) * h) ** 2
                    + ((hand_lms[a].z - hand_lms[b].z) * w) ** 2
                )
                for a, b in PALM_SPAN_EDGES
            ) / len(PALM_SPAN_EDGES)
            palm_norm = [round(avg_x, 3), round(avg_y, 3), round(avg_z, 3)]
            wrist_px = list(pts[0])

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox_px = [min(xs), min(ys), max(xs), max(ys)]

            # Draw palm connections in light grey
            for i_a, i_b in PALM_BRIDGES:
                cv2.line(annotated, pts[i_a], pts[i_b], (200, 200, 200), 2, cv2.LINE_AA)

            # Draw each independent finger as segmented lines (Green = Extended, Red = Curled)
            fingertip_px = {}
            for fname, chain in FINGER_CHAINS.items():
                is_open = finger_states[fname]
                line_color = (0, 255, 0) if is_open else (0, 0, 255)

                # Draw bone segments along the finger
                for idx1, idx2 in zip(chain[:-1], chain[1:]):
                    cv2.line(annotated, pts[idx1], pts[idx2], line_color, 3, cv2.LINE_AA)

                # Draw joint dots along the finger
                for idx in chain[1:]:
                    cv2.circle(annotated, pts[idx], 4, (255, 255, 255), cv2.FILLED)
                    cv2.circle(annotated, pts[idx], 4, line_color, 1)

                # Highlight fingertip and label
                tip_idx = chain[-1]
                cx, cy = pts[tip_idx]
                fingertip_px[fname] = [cx, cy]
                cv2.circle(annotated, (cx, cy), 7, line_color, cv2.FILLED)
                cv2.putText(
                    annotated,
                    f"{fname[0].upper()}:{is_open}",
                    (cx - 15, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    line_color,
                    2,
                    cv2.LINE_AA,
                )

            # Draw wrist point
            cv2.circle(annotated, pts[0], 6, (255, 200, 0), cv2.FILLED)

            # Draw palm center marker & coordinates
            px_c, py_c = palm_px
            cv2.circle(annotated, (px_c, py_c), 7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.line(annotated, (px_c - 10, py_c), (px_c + 10, py_c), (255, 255, 0), 2, cv2.LINE_AA)
            cv2.line(annotated, (px_c, py_c - 10), (px_c, py_c + 10), (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"Pos:({px_c},{py_c})",
                (px_c + 12, py_c + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Highlight thumb-index pinch circle for OK_SIGN or PINCH
            if gesture in ("OK_SIGN", "PINCH"):
                pinch_mid = (
                    (pts[4][0] + pts[8][0]) // 2,
                    (pts[4][1] + pts[8][1]) // 2,
                )
                cv2.circle(annotated, pinch_mid, 14, (255, 0, 255), 2, cv2.LINE_AA)

            # Draw bounding box & high-contrast gesture banner
            gesture_color = GESTURE_COLORS.get(gesture, (255, 255, 255))
            cv2.rectangle(
                annotated,
                (bbox_px[0], bbox_px[1]),
                (bbox_px[2], bbox_px[3]),
                gesture_color,
                1,
                cv2.LINE_AA,
            )

            banner = f"[{label}] {gesture} ({px_c}, {py_c})"
            (tw, th), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bx = max(4, min(bbox_px[0], w - tw - 12))
            by = max(th + 10, bbox_px[1] - 8)
            cv2.rectangle(
                annotated,
                (bx - 4, by - th - 6),
                (bx + tw + 6, by + baseline + 2),
                (20, 20, 20),
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                banner,
                (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                gesture_color,
                2,
                cv2.LINE_AA,
            )

            hands_payload.append({
                "hand": label,
                "gesture": gesture,
                "fingers": finger_states,
                "count": sum(finger_states.values()),
                "palm_px": palm_px,
                "palm_px_f": palm_px_f,
                "palm_span_px": round(palm_span_px, 2),
                "palm_norm": palm_norm,
                "wrist_px": wrist_px,
                "bbox_px": bbox_px,
                "tips_px": fingertip_px,
            })

        return hands_payload

    def _log_terminal_debug(self, hands_payload: list[dict]):
        """Log detected hand configurations and positions to the terminal/ROS log."""
        if not self.debug_terminal:
            return

        now = time.monotonic()
        signature = tuple((h["hand"], h["gesture"], h["count"]) for h in hands_payload)

        if not hands_payload:
            if self._last_debug_signature:
                self.get_logger().info("[HAND DEBUG] No hands detected.")
                self._last_debug_signature = ()
                self._last_debug_log_time = now
            return

        if (
            signature == self._last_debug_signature
            and (now - self._last_debug_log_time) < self._debug_period
        ):
            return

        lines = [f"[HAND DEBUG] Detected {len(hands_payload)} hand(s):"]
        for idx, h in enumerate(hands_payload, start=1):
            f = h["fingers"]
            fingers_str = (
                f"T:{f['thumb']} I:{f['index']} M:{f['middle']} "
                f"R:{f['ring']} P:{f['pinky']} ({h['count']}/5)"
            )
            tips = h["tips_px"]
            tips_str = (
                f"T{tuple(tips['thumb'])} I{tuple(tips['index'])} "
                f"M{tuple(tips['middle'])} R{tuple(tips['ring'])} P{tuple(tips['pinky'])}"
            )
            lines.append(
                f"  #{idx} [{h['hand']}] Gesture={h['gesture']} | {fingers_str}\n"
                f"      Palm_px={tuple(h['palm_px'])} Palm_norm={tuple(h['palm_norm'])} "
                f"Wrist_px={tuple(h['wrist_px'])} BBox={h['bbox_px']}\n"
                f"      Tips_px: {tips_str}"
            )

        self.get_logger().info("\n".join(lines))
        self._last_debug_signature = signature
        self._last_debug_log_time = now

    def raw_image_callback(self, msg: Image):
        # Run MediaPipe on the clean camera frame
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._latest_result = self.detector.detect(mp_image)

        annotated = frame.copy()
        hands_payload = self._draw_hands(annotated, self._latest_result)
        self._log_terminal_debug(hands_payload)

        # Publish JSON state of fingers, gestures, and hand positions
        state_msg = String()
        state_msg.data = json.dumps(hands_payload)
        self.pub_state.publish(state_msg)

        # Publish MediaPipe-only view on /mediapipe/image_annotated
        mp_out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        mp_out.header = msg.header
        self.pub_mp.publish(mp_out)

    def yolo_image_callback(self, msg: Image):
        # Draw the latest detected finger lines and gesture labels on top of YOLO's bounding boxes
        annotated = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._draw_hands(annotated, self._latest_result)

        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        self.pub_yolo_combined.publish(out_msg)

    def destroy_node(self):
        self.detector.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandNode()
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
