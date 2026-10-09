"""Tests of the safety rules of real_arm_server, with a fake bus (no hardware).

    python3 -m unittest discover -s tools
"""
import json
import socket
import threading
import time
import unittest

import real_arm_server as srv


class FakeBus:
    def __init__(self):
        self.pos = {"shoulder_pan": 0.0, "shoulder_lift": 0.1, "elbow_flex": 0.2,
                    "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 50.0}
        self.writes = []
        self.torque = False

    def sync_read(self, name):
        assert name == "Present_Position"
        return dict(self.pos)

    def sync_write(self, name, values):
        assert name == "Goal_Position"
        self.writes.append(dict(values))

    def enable_torque(self, *a, **k):
        self.torque = True

    def disable_torque(self, *a, **k):
        self.torque = False


class BridgeTests(unittest.TestCase):
    def test_read_only_by_default(self):
        bus = FakeBus()
        b = srv.ArmBridge(bus)
        self.assertEqual(b.handle({"cmd": "read"})["obs"]["gripper.pos"], 50.0)
        r = b.handle({"cmd": "torque", "on": True})
        self.assertFalse(r["ok"])
        self.assertFalse(bus.torque)
        self.assertEqual(bus.writes, [])

    def test_act_ignored_while_torque_off(self):
        bus = FakeBus()
        b = srv.ArmBridge(bus, allow_motion=True)
        r = b.handle({"cmd": "act", "action": {"shoulder_pan.pos": 1.0}})
        self.assertTrue(r["ok"])
        self.assertEqual(r["applied"], {})
        self.assertEqual(bus.writes, [])

    def test_enable_torque_holds_present_position_first(self):
        bus = FakeBus()
        b = srv.ArmBridge(bus, allow_motion=True)
        self.assertTrue(b.handle({"cmd": "torque", "on": True})["ok"])
        self.assertTrue(bus.torque)
        self.assertEqual(bus.writes[0], bus.pos)  # goal = present, written before enabling

    def test_goal_is_clipped_to_max_step(self):
        bus = FakeBus()
        b = srv.ArmBridge(bus, allow_motion=True, max_step=0.05, max_step_gripper=10.0)
        b.handle({"cmd": "torque", "on": True})
        r = b.handle({"cmd": "act", "action": {"shoulder_pan.pos": 1.0, "elbow_flex.pos": -3.0,
                                               "gripper.pos": 100.0, "bogus.pos": 9.0}})
        self.assertAlmostEqual(r["applied"]["shoulder_pan.pos"], 0.05)
        self.assertAlmostEqual(r["applied"]["elbow_flex.pos"], 0.15)
        self.assertAlmostEqual(r["applied"]["gripper.pos"], 60.0)
        self.assertNotIn("bogus.pos", r["applied"])

    def test_small_goal_passes_unchanged(self):
        bus = FakeBus()
        b = srv.ArmBridge(bus, allow_motion=True)
        b.handle({"cmd": "torque", "on": True})
        r = b.handle({"cmd": "act", "action": {"shoulder_pan.pos": 0.03}})
        self.assertAlmostEqual(r["applied"]["shoulder_pan.pos"], 0.03)

    def test_unknown_command_and_bus_error_do_not_raise(self):
        b = srv.ArmBridge(FakeBus())
        self.assertFalse(b.handle({"cmd": "nope"})["ok"])
        b.bus.sync_read = lambda name: (_ for _ in ()).throw(IOError("serial gone"))
        self.assertFalse(b.handle({"cmd": "read"})["ok"])

    def test_shutdown_releases_torque_only_if_we_enabled_it(self):
        bus = FakeBus()
        b = srv.ArmBridge(bus, allow_motion=True)
        b.shutdown()
        self.assertEqual(bus.torque, False)
        b.handle({"cmd": "torque", "on": True})
        b.shutdown()
        self.assertFalse(bus.torque)


class SocketTests(unittest.TestCase):
    def _start(self, bridge, watchdog=0.3):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        threading.Thread(target=srv.serve, args=(bridge, "127.0.0.1", port, watchdog), daemon=True).start()
        for _ in range(50):
            try:
                return port, socket.create_connection(("127.0.0.1", port))
            except ConnectionRefusedError:
                time.sleep(0.05)
        raise RuntimeError("server did not start")

    @staticmethod
    def _call(sock, msg):
        sock.sendall((json.dumps(msg) + "\n").encode())
        return json.loads(sock.makefile().readline())

    def test_roundtrip_and_watchdog_freeze(self):
        bus = FakeBus()
        bridge = srv.ArmBridge(bus, allow_motion=True)
        _, sock = self._start(bridge)
        self.assertTrue(self._call(sock, {"cmd": "ping"})["ok"])
        self.assertTrue(self._call(sock, {"cmd": "torque", "on": True})["ok"])
        n = len(bus.writes)
        time.sleep(0.8)  # silence > watchdog
        self.assertGreater(len(bus.writes), n)
        self.assertEqual(bus.writes[-1], bus.pos)  # frozen at present position
        self.assertTrue(bus.torque)  # torque kept, arm not dropped

    def test_bad_json_is_reported(self):
        _, sock = self._start(srv.ArmBridge(FakeBus()))
        sock.sendall(b"not json\n")
        self.assertFalse(json.loads(sock.makefile().readline())["ok"])


if __name__ == "__main__":
    unittest.main()
