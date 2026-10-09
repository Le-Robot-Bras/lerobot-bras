"""remote_arm.py - client of tools/real_arm_server.py.

Same API as SO101Sim / SO101Follower (connect / get_observation / send_action /
disconnect / is_connected), so driver_node does not care which one runs. The arm
is on the host (macOS + Docker Desktop cannot pass USB to a container).

Units: radians for the 5 arm joints, 0-100 for the gripper.
"""
import json
import socket

DEFAULT_REMOTE_PORT = 5555


class SO101Remote:
    def __init__(self, host: str, port: int = DEFAULT_REMOTE_PORT, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._file = None

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    def connect(self, calibrate: bool = True) -> None:  # same signature as the real arm
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._file = self._sock.makefile("rb")
        self._call({"cmd": "ping"})

    def disconnect(self) -> None:
        # The server freezes the arm where it is; it does not drop it.
        if self._sock is not None:
            self._sock.close()
        self._sock = self._file = None

    def get_observation(self) -> dict:
        return self._call({"cmd": "read"})["obs"]

    def send_action(self, action: dict) -> dict:
        return self._call({"cmd": "act", "action": action})["applied"]

    def info(self) -> dict:
        return self._call({"cmd": "info"})

    def enable_torque(self) -> None:
        self._call({"cmd": "torque", "on": True})

    def disable_torque(self) -> None:
        self._call({"cmd": "torque", "on": False})

    def _call(self, msg: dict) -> dict:
        if self._sock is None:
            raise RuntimeError("SO101Remote not connected: call connect() first.")
        self._sock.sendall((json.dumps(msg) + "\n").encode())
        line = self._file.readline()
        if not line:
            raise ConnectionError("real_arm_server closed the connection")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"real_arm_server: {reply.get('error')}")
        return reply
