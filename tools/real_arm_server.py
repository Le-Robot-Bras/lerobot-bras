#!/usr/bin/env python3
"""real_arm_server.py - serves the REAL SO-ARM101 over TCP.

Why: Docker Desktop (macOS) cannot pass a USB serial port to a container. This
server runs on the host, owns the arm, and the `control` container talks to it
through so101_driver/remote_arm.py (same API as SO101Sim / SO101Follower).

    uv run --with ./core/lerobot_min tools/real_arm_server.py \
        --port /dev/cu.usbmodemXXXX \
        --calibration ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json

Safety:
  - Read-only by default: positions can be read, nothing can move the arm.
  - Motion needs --allow-motion AND an explicit {"cmd": "torque", "on": true}.
  - Enabling torque first sets the goal to the present position (no jump).
  - Every goal is clipped to +-max-step of the present position (rate limit).
  - If the client goes silent for --watchdog seconds, or disconnects, the arm is
    frozen where it is (goal = present). Torque is only released on explicit
    request or when this server exits.
  - Listens on 127.0.0.1 only (Docker Desktop reaches it via host.docker.internal).

Protocol: one JSON object per line, one JSON reply per line.
    {"cmd": "ping"}
    {"cmd": "info"}
    {"cmd": "read"}                          -> {"obs": {"shoulder_pan.pos": rad, ..., "gripper.pos": 0-100}}
    {"cmd": "torque", "on": true|false}
    {"cmd": "act", "action": {"shoulder_pan.pos": 0.1, ...}}
"""
import argparse
import json
import logging
import socket
import time
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555
GRIPPER = "gripper"

log = logging.getLogger("real_arm_server")


class ArmBridge:
    """Turns JSON requests into bus operations, with the safety rules above."""

    def __init__(self, bus, allow_motion=False, max_step=0.05, max_step_gripper=10.0):
        self.bus = bus
        self.allow_motion = allow_motion
        self.max_step = max_step  # rad per command, arm joints
        self.max_step_gripper = max_step_gripper  # % per command
        self.torque_on = False

    def _present(self) -> dict[str, float]:
        return {m: float(v) for m, v in self.bus.sync_read("Present_Position").items()}

    def handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        try:
            if cmd == "ping":
                return {"ok": True}
            if cmd == "info":
                return {
                    "ok": True,
                    "allow_motion": self.allow_motion,
                    "torque": self.torque_on,
                    "max_step": self.max_step,
                    "max_step_gripper": self.max_step_gripper,
                }
            if cmd == "read":
                obs = {f"{m}.pos": v for m, v in self._present().items()}
                return {"ok": True, "obs": obs, "torque": self.torque_on}
            if cmd == "torque":
                return self._set_torque(bool(msg.get("on")))
            if cmd == "act":
                return self._act(msg.get("action", {}))
            return {"ok": False, "error": f"unknown cmd: {cmd!r}"}
        except Exception as e:  # keep serving: a bad request must not kill the arm link
            log.exception("request failed: %s", msg)
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def _set_torque(self, on: bool) -> dict:
        if on:
            if not self.allow_motion:
                return {"ok": False, "error": "motion is disabled: start the server with --allow-motion"}
            self.bus.sync_write("Goal_Position", self._present())  # hold here, do not jump
            self.bus.enable_torque()
            self.torque_on = True
            log.warning("TORQUE ENABLED")
        else:
            self.bus.disable_torque()
            self.torque_on = False
            log.warning("torque disabled (arm is limp)")
        return {"ok": True, "torque": self.torque_on}

    def _act(self, action: dict) -> dict:
        if not self.torque_on:  # not an error: the driver keeps streaming commands
            return {"ok": True, "applied": {}, "torque": False}

        present = self._present()
        goal = {}
        for key, value in action.items():
            motor = key.removesuffix(".pos")
            if motor not in present:
                continue
            step = self.max_step_gripper if motor == GRIPPER else self.max_step
            goal[motor] = min(max(float(value), present[motor] - step), present[motor] + step)
        if goal:
            self.bus.sync_write("Goal_Position", goal)
        return {"ok": True, "applied": {f"{m}.pos": v for m, v in goal.items()}, "torque": True}

    def freeze(self):
        """Stop where we are: goal = present. Torque stays on."""
        if not self.torque_on:
            return
        try:
            self.bus.sync_write("Goal_Position", self._present())
        except Exception:
            log.exception("freeze failed")

    def shutdown(self):
        if self.torque_on:
            log.warning("exiting: releasing torque, the arm will relax")
            self.bus.disable_torque(num_retry=5)
            self.torque_on = False


def serve(bridge: ArmBridge, host: str, port: int, watchdog: float):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    log.info("listening on %s:%d (allow_motion=%s)", host, port, bridge.allow_motion)

    while True:
        conn, addr = srv.accept()
        log.info("client connected: %s", addr)
        try:
            _serve_client(bridge, conn, watchdog)
        except (ConnectionError, OSError) as e:
            log.warning("client link lost: %s", e)
        finally:
            conn.close()
            bridge.freeze()
            log.info("client gone")


def _serve_client(bridge: ArmBridge, conn: socket.socket, watchdog: float):
    conn.settimeout(0.2)
    buf = b""
    last_msg = time.monotonic()
    frozen = False
    while True:
        try:
            data = conn.recv(4096)
        except socket.timeout:
            if bridge.torque_on and not frozen and time.monotonic() - last_msg > watchdog:
                log.warning("watchdog: no message for %.1fs, freezing the arm", watchdog)
                bridge.freeze()
                frozen = True
            continue
        if not data:
            return
        last_msg, frozen = time.monotonic(), False
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try:
                reply = bridge.handle(json.loads(line))
            except json.JSONDecodeError as e:
                reply = {"ok": False, "error": f"bad JSON: {e}"}
            conn.sendall((json.dumps(reply) + "\n").encode())


def open_arm(port: str, calibration: str):
    """Opens the serial bus WITHOUT touching torque (SO101Follower.connect() would enable it).

    Returns the robot, not just its bus: Robot.__del__ disconnects the bus (and disables
    torque) when the robot is garbage-collected, so the caller must keep a reference to it.
    """
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    calibration = Path(calibration).expanduser()
    robot = SO101Follower(
        SO101FollowerConfig(
            port=port,
            id=calibration.stem,
            calibration_dir=calibration.parent,
            use_radians=True,
        )
    )
    robot.bus.connect()  # pings the 6 motors, reads firmware versions: read-only
    if not robot.bus.is_calibrated:
        robot.bus.disconnect(disable_torque=False)
        raise SystemExit(
            f"{calibration} does not match the motors: wrong file for this arm? "
            "Re-run lerobot-calibrate."
        )
    return robot


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="serial port of the arm, e.g. /dev/cu.usbmodemXXXX")
    p.add_argument("--calibration", required=True, help="calibration JSON (from lerobot-calibrate)")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--listen", type=int, default=DEFAULT_PORT, help="TCP port (default %(default)s)")
    p.add_argument("--allow-motion", action="store_true", help="allow torque and motion (default: read-only)")
    p.add_argument("--max-step", type=float, default=0.05, help="max rad per command, arm joints")
    p.add_argument("--max-step-gripper", type=float, default=10.0, help="max %% per command, gripper")
    p.add_argument("--watchdog", type=float, default=1.0, help="freeze the arm after this many silent seconds")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    robot = open_arm(args.port, args.calibration)  # keep `robot` alive for the whole run
    bus = robot.bus
    bridge = ArmBridge(bus, args.allow_motion, args.max_step, args.max_step_gripper)
    log.info("arm connected on %s (%s)", args.port, "motion ALLOWED" if args.allow_motion else "READ-ONLY")
    try:
        serve(bridge, args.host, args.listen, args.watchdog)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.shutdown()
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
