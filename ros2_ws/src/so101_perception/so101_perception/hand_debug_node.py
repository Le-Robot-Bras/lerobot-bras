#!/usr/bin/env python3
"""Interactive terminal debug monitor for hand gestures and positions."""
import json
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


class HandDebugNode(Node):
    """Subscribe to /hand/fingers_state and render a live terminal debug view."""

    def __init__(self):
        super().__init__("hand_debug")
        self._is_tty = sys.stdout.isatty()
        self._msg_count = 0
        self._last_render = 0.0
        self._last_empty_logged = False
        self._min_period = 0.15  # up to ~6.6 Hz refresh for smooth readability

        self.create_subscription(String, "/hand/fingers_state", self._cb_state, 10)
        print(
            "=== Hand Gesture & Position Terminal Debugger ===\n"
            "Listening on /hand/fingers_state (Ctrl+C to exit)...",
            flush=True,
        )

    def _format_dashboard(self, hands: list[dict]) -> str:
        ts = time.strftime("%H:%M:%S")
        header = (
            f"=== Hand Debug [{ts}] | msgs={self._msg_count} | "
            f"hands={len(hands)} ==="
        )
        if not hands:
            return f"{header}\n  (no hands currently detected)"

        lines = [header]
        for idx, h in enumerate(hands, start=1):
            hand_label = h.get("hand", "Unknown")
            gesture = h.get("gesture", "UNKNOWN")
            count = h.get("count", 0)
            fingers = h.get("fingers", {})
            palm_px = h.get("palm_px", ["?", "?"])
            palm_norm = h.get("palm_norm", ["?", "?", "?"])
            wrist_px = h.get("wrist_px", ["?", "?"])
            bbox_px = h.get("bbox_px", ["?", "?", "?", "?"])
            tips_px = h.get("tips_px", {})

            finger_flags = "  ".join(
                f"{fname.upper()}:{'UP  ' if fingers.get(fname, 0) else 'DOWN'}"
                for fname in FINGER_ORDER
            )
            tips_line = "  ".join(
                f"{fname[0].upper()}:{tuple(tips_px[fname])}"
                for fname in FINGER_ORDER
                if fname in tips_px
            )

            lines.append(
                f"\n  Hand #{idx} [{hand_label}] -> CONFIG: {gesture} ({count}/5 fingers)"
            )
            lines.append(f"    Fingers   : {finger_flags}")
            lines.append(
                f"    Palm Pos  : px=({palm_px[0]}, {palm_px[1]})  "
                f"norm=({palm_norm[0]}, {palm_norm[1]}, {palm_norm[2]})"
            )
            lines.append(
                f"    Wrist/Box : wrist_px=({wrist_px[0]}, {wrist_px[1]})  "
                f"bbox_px={bbox_px}"
            )
            lines.append(f"    Tips (px) : {tips_line}")

        return "\n".join(lines)

    def _cb_state(self, msg: String):
        self._msg_count += 1
        now = time.monotonic()
        if now - self._last_render < self._min_period:
            return

        try:
            hands = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Received invalid JSON on /hand/fingers_state")
            return

        if not self._is_tty and not hands and self._last_empty_logged:
            return

        output = self._format_dashboard(hands)
        if self._is_tty:
            sys.stdout.write("\033[2J\033[H" + output + "\n")
            sys.stdout.flush()
        else:
            print(output, flush=True)

        self._last_render = now
        self._last_empty_logged = not bool(hands)


def main(args=None):
    rclpy.init(args=args)
    node = HandDebugNode()
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
