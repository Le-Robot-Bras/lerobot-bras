#!/usr/bin/env python3
"""demo_loop.py - live demo: N pick & place in a row, a new random cube each time (sim).

Runs in the control container, next to the running stack (driver + brain + perception):

    docker compose exec control bash -lc "source /ros2_ws/install/setup.bash && \\
        ros2 run so101_brain demo_loop --runs 5"

Each run: /sim/reset_object (driver, sim only) -> /brain/start -> wait for /brain_state
"DONE ..." (or a timeout) -> log the result and the time. A summary is printed at the end.
Watch it in RViz (noVNC): the cube follows the gripper, targets and state are drawn.
"""
import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class DemoLoop(Node):
    def __init__(self):
        super().__init__("so101_demo_loop")
        self.state = ""
        self.create_subscription(String, "/brain_state", self._cb_state, 10)
        self.reset = self.create_client(Trigger, "/sim/reset_object")
        self.start = self.create_client(Trigger, "/brain/start")

    def _cb_state(self, msg):
        self.state = msg.data

    def call(self, client, timeout=10.0):
        if not client.wait_for_service(timeout_sec=timeout):
            return False, f"service {client.srv_name} not available"
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() is None:
            return False, f"{client.srv_name}: no answer"
        return future.result().success, future.result().message

    def wait_state(self, prefixes, timeout):
        """Spin until /brain_state starts with one of `prefixes`. Returns the state or None."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state.startswith(prefixes):
                return self.state
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=90.0, help="s per pick & place")
    parser.add_argument("--pause", type=float, default=2.0, help="s between runs")
    args = parser.parse_args()

    rclpy.init()
    node = DemoLoop()
    log = node.get_logger()
    results = []
    try:
        # The brain must be idle or done before a new run.
        if node.wait_state(("IDLE", "DONE"), timeout=30.0) is None:
            log.error(f"brain not idle (state: '{node.state}'): call /brain/stop first")
            return
        for run in range(1, args.runs + 1):
            ok, msg = node.call(node.reset)
            if not ok:
                log.error(f"run {run}: reset failed: {msg}")
                break
            time.sleep(0.5)  # let the perception see the new scene
            t0 = time.monotonic()
            ok, msg = node.call(node.start)
            if not ok:
                log.error(f"run {run}: /brain/start refused: {msg}")
                break
            node.state = ""
            node.wait_state(("HOME",), timeout=5.0)  # wait until it actually left DONE
            final = node.wait_state(("DONE",), timeout=args.timeout)
            dt = time.monotonic() - t0
            success = final is not None
            results.append((success, dt))
            log.info(f"run {run}/{args.runs}: {'SUCCESS' if success else 'TIMEOUT'} in {dt:.1f} s "
                     f"(brain: {node.state})")
            if not success:
                node.call(node.create_client(Trigger, "/brain/stop"))
            time.sleep(args.pause)
    except KeyboardInterrupt:
        pass
    finally:
        if results:
            n_ok = sum(r[0] for r in results)
            times = [r[1] for r in results if r[0]]
            mean = sum(times) / len(times) if times else float("nan")
            log.info(f"demo summary: {n_ok}/{len(results)} cubes placed, mean time {mean:.1f} s")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
