"""rosbridge.py - streams the dance state to ROS 2 (RViz) without touching the arm link.

The real arm server accepts a single client (the dance), so ROS can't read the arm itself.
Instead the dance serves its state as JSON lines on a TCP port (default 127.0.0.1:5600):
    {"t": 12.34, "names": [...], "target": [...], "measured": [...] | null, "label": "...",
     "style": "...", "genre": "..."}
Joints in radians, gripper in % (0-100), like real_arm_server. The ROS node
`ros2 run so101_driver dance_bridge` (in the control container) connects to it through
host.docker.internal and republishes /joint_states (-> robot_state_publisher -> TF -> RViz)
and /dance/state. Only the latest state is sent: a slow client never slows the dance.
"""
import json
import logging
import socket
import threading
import time

log = logging.getLogger("dancelive.rosbridge")


class StateStream:
    def __init__(self, host="127.0.0.1", port=5600, hz=30.0):
        self.period = 1.0 / hz
        self._lock = threading.Lock()
        self._msg = None
        self._seq = 0
        self._stop = threading.Event()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(2)
        self._srv.settimeout(0.5)
        self.url = f"{host}:{port}"
        threading.Thread(target=self._accept, daemon=True, name="rosbridge-accept").start()

    def update(self, target, measured=None, label="", style="", genre=""):
        """Called from the control loop: just stores the latest state (no I/O)."""
        names = list(target.keys())
        msg = {"t": time.time(), "names": names,
               "target": [float(target[n]) for n in names],
               "measured": None if measured is None else [float(measured.get(n, target[n])) for n in names],
               "label": label, "style": style, "genre": genre}
        with self._lock:
            self._msg, self._seq = msg, self._seq + 1

    def _accept(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            log.info("ROS bridge client connected: %s", addr)
            threading.Thread(target=self._serve, args=(conn,), daemon=True, name="rosbridge-client").start()

    def _serve(self, conn):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sent = -1
        try:
            while not self._stop.is_set():
                with self._lock:
                    msg, seq = self._msg, self._seq
                if msg is not None and seq != sent:
                    conn.sendall((json.dumps(msg) + "\n").encode())
                    sent = seq
                time.sleep(self.period)
        except OSError:
            pass
        finally:
            conn.close()

    def close(self):
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass


def open_stream(args):
    """The ROS state stream if --ros-bridge, else None (a busy port only disables it)."""
    if not getattr(args, "ros_bridge", False):
        return None
    try:
        s = StateStream(args.ros_bridge_host, args.ros_bridge_port)
    except OSError as e:
        print(f"ROS bridge disabled: port {args.ros_bridge_port}: {e}")
        return None
    print(f"ROS bridge: dance state on {s.url} (in Docker: ros2 run so101_driver dance_bridge)")
    return s
