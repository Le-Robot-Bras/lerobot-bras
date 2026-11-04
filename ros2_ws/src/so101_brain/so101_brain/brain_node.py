#!/usr/bin/env python3
"""brain_node.py - intelligence of the arm: kinematics and pick & place.

Never talks to the backend: reads /joint_states, commands through /joint_command.

  Subscribes  /joint_states, /object_position, /drop_box_position
  Publishes   /joint_command        (20 Hz while running: rad + gripper 0-100 %)
              /end_effector_pose    (TCP pose from forward kinematics, for RViz)
              /brain_state          (std_msgs/String: STATE - status)
              /brain/markers        (visualization_msgs/MarkerArray, RViz: the cube where the brain
                                    believes it is - on the table, in the gripper, in the box -,
                                    the planned pre-grasp / drop points, and the state as text)
  Services    /brain/start, /brain/stop   (std_srvs/Trigger)  pick & place
              /go_to_target               (so101_interfaces/GoToTarget) move the TCP to an
                                          XYZ (position-only IK), only when the brain is idle

The logic lives in pick_place.py / kinematics.py (pure Python, also run without
ROS by tools/eval_pick_place.py). Perception never blocks the loop: the state
machine always uses the latest estimate it received.
"""
from pathlib import Path

import numpy as np
import rclpy
import rclpy.node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from so101_interfaces.srv import GoToTarget
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from so101_brain.kinematics import ARM_JOINT_NAMES, ArmKinematics
from so101_brain.pick_place import Observation, PickPlace, PickPlaceConfig, State

BASE_FRAME = "base_link"
# world -> base_link is the identity (robot base at the MuJoCo origin, see viz.launch.py).
ACCEPTED_FRAMES = ("", BASE_FRAME, "world")
GRIPPER_RANGE_RAD = (-0.174533, 1.74533)   # URDF gripper joint, 0 % .. 100 %
LOOP_HZ = 20.0
MARKER_EVERY = 2        # markers at 10 Hz
CUBE_SIZE = 0.05
STATE_COLORS = {        # text colour per phase (r, g, b)
    "IDLE": (0.8, 0.8, 0.8), "DONE": (0.2, 1.0, 0.3), "RETRY": (1.0, 0.5, 0.1),
}


def gripper_rad_to_pct(rad: float) -> float:
    lo, hi = GRIPPER_RANGE_RAD
    return float(np.clip((rad - lo) / (hi - lo) * 100.0, 0.0, 100.0))


def yaw_of(msg: PoseStamped) -> float:
    q = msg.pose.orientation
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


class BrainNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("so101_brain")
        urdf = Path(get_package_share_directory("so101_description")) / "urdf" / "so101.urdf"
        self._kin = ArmKinematics(urdf)

        config = PickPlaceConfig()
        # A few useful knobs as ROS parameters (the rest: PickPlaceConfig).
        for name in ("gripper_open", "drop_height", "lift_height", "joint_speed", "cart_speed",
                     "max_attempts"):
            default = getattr(config, name)
            setattr(config, name, type(default)(self.declare_parameter(name, default).value))
        self._auto_start = bool(self.declare_parameter("auto_start", False).value)
        self._pp = PickPlace(self._kin, config, log=self._log)

        self._joints = None
        self._pub_command = self.create_publisher(JointState, "/joint_command", 10)
        self._pub_ee_pose = self.create_publisher(PoseStamped, "/end_effector_pose", 10)
        self._pub_state = self.create_publisher(String, "/brain_state", 10)
        self._pub_markers = self.create_publisher(MarkerArray, "/brain/markers", 2)
        self._tick = 0
        self.create_subscription(JointState, "/joint_states", self._cb_joint_states, 10)
        self.create_subscription(PoseStamped, "/object_position", self._cb_object, 10)
        self.create_subscription(PoseStamped, "/drop_box_position", self._cb_box, 10)
        self.create_service(Trigger, "/brain/start", self._cb_start)
        self.create_service(Trigger, "/brain/stop", self._cb_stop)
        self.create_service(GoToTarget, "/go_to_target", self._cb_go_to_target)
        self.create_timer(1.0 / LOOP_HZ, self._loop)
        self.get_logger().info("Brain node ready. Start with: ros2 service call /brain/start std_srvs/srv/Trigger")

    # --------------------------------------------------------------- helpers

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _log(self, level, msg):
        # One call site per severity: rclpy refuses to change the severity of a call site.
        if level == "error":
            self.get_logger().error(msg)
        elif level == "warn":
            self.get_logger().warning(msg)
        else:
            self.get_logger().info(msg)

    def _observation(self, msg: PoseStamped):
        if msg.header.frame_id not in ACCEPTED_FRAMES:
            self.get_logger().warning(f"Ignoring pose in frame '{msg.header.frame_id}'",
                                      throttle_duration_sec=10.0)
            return None
        p = msg.pose.position
        # Stamp with the reception time: perception is fast, and it avoids clock skew between containers.
        return Observation(np.array([p.x, p.y, p.z]), yaw_of(msg), self._now())

    # ------------------------------------------------------------- callbacks

    def _cb_joint_states(self, msg: JointState):
        joints = dict(zip(msg.name, msg.position))
        if not all(n in joints for n in ARM_JOINT_NAMES):
            return
        self._joints = joints
        gripper = gripper_rad_to_pct(joints["gripper"]) if "gripper" in joints else None
        self._pp.update_joints({n: joints[n] for n in ARM_JOINT_NAMES}, gripper)
        self._publish_end_effector_pose()
        if self._auto_start and self._pp.state == State.IDLE:
            ok, why = self._pp.start(self._now())
            self._auto_start = not ok
            self.get_logger().info(f"auto_start: {why}")

    def _cb_object(self, msg):
        obs = self._observation(msg)
        if obs is not None:
            self._pp.update_object(obs)

    def _cb_box(self, msg):
        obs = self._observation(msg)
        if obs is not None:
            self._pp.update_box(obs)

    def _cb_start(self, request, response):
        response.success, response.message = self._pp.start(self._now())
        return response

    def _cb_stop(self, request, response):
        response.success, response.message = self._pp.stop(self._now())
        return response

    def _cb_go_to_target(self, request, response):
        """Move the TCP to request.target_pose position (orientation ignored)."""
        if self._pp.state not in (State.IDLE, State.DONE):
            response.success, response.message = False, f"brain busy ({self._pp.state.value}), call /brain/stop"
            return response
        if self._joints is None:
            response.success, response.message = False, "no /joint_states yet"
            return response
        frame = request.target_pose.header.frame_id
        if frame not in ACCEPTED_FRAMES:
            response.success, response.message = False, f"Unsupported frame '{frame}', use '{BASE_FRAME}'"
            return response
        p = request.target_pose.pose.position
        q0 = self._kin.q_from_joints(self._joints)
        res = self._kin.solve_global([p.x, p.y, p.z], q0, frame="tcp")
        if not res.ok(1e-3):
            response.success = False
            response.message = (f"Target ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) unreachable "
                                f"(residual {res.position_error * 1000:.1f} mm)")
            return response
        self._publish_command(self._kin.joints_from_q(res.q), None)
        response.success = True
        response.message = "Joint command sent: " + ", ".join(
            f"{n}={v:.3f}" for n, v in self._kin.joints_from_q(res.q).items())
        return response

    # ------------------------------------------------------------------ loop

    def _loop(self):
        command = self._pp.step(self._now())
        if command is not None:
            self._publish_command(command.joints, command.gripper)
        self._pub_state.publish(String(data=f"{self._pp.state.value} - {self._pp.status}"))
        self._tick += 1
        if self._tick % MARKER_EVERY == 0:
            self._publish_markers()

    def _marker(self, ns, mid, mtype, xyz, scale, rgba, yaw=0.0):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in xyz)
        m.pose.orientation.z, m.pose.orientation.w = float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
        m.color.r, m.color.g, m.color.b, m.color.a = (float(v) for v in rgba)
        m.lifetime.nanosec = 500_000_000  # vanishes if the brain stops publishing it
        return m

    def _publish_markers(self):
        markers = MarkerArray()
        where, pos = self._pp.belief()
        if pos is not None:
            alpha = 1.0 if where in ("held", "placed") else 0.9
            markers.markers.append(self._marker("brain_cube", 0, Marker.CUBE, pos, (CUBE_SIZE,) * 3,
                                                (0.85, 0.05, 0.2, alpha), self._pp.cube_yaw()))
            # No spaces: RViz (software rendering in noVNC) draws them very wide.
            label = {"table": "cube:seen", "held": "cube:held", "placed": "cube:placed"}[where]
            markers.markers.append(self._marker("brain_cube", 1, Marker.TEXT_VIEW_FACING,
                                                np.asarray(pos) + [0.0, 0.0, 0.05], (0, 0, 0.018),
                                                (1, 1, 1, 0.9)))
            markers.markers[-1].text = label
        colors = {"pregrasp": (0.2, 0.6, 1.0, 0.8), "grasp": (0.2, 0.9, 1.0, 0.8), "drop": (1.0, 0.8, 0.1, 0.8)}
        for i, (name, p) in enumerate(self._pp.targets().items()):
            markers.markers.append(self._marker("brain_targets", i, Marker.SPHERE, p, (0.012,) * 3, colors[name]))
        state = self._pp.state.value
        text = self._marker("brain_state", 0, Marker.TEXT_VIEW_FACING, (0.15, -0.15, 0.42), (0, 0, 0.035),
                            (*STATE_COLORS.get(state, (1.0, 1.0, 0.3)), 1.0))
        text.text = f"{state}_attempt{self._pp.attempt + 1}" if state in ("APPROACH", "DESCEND", "GRASP", "RETRY") else state
        markers.markers.append(text)
        self._pub_markers.publish(markers)

    def _publish_command(self, joints: dict, gripper_pct):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joints)
        msg.position = [float(v) for v in joints.values()]
        if gripper_pct is not None:
            msg.name.append("gripper")
            msg.position.append(float(np.clip(gripper_pct, 0.0, 100.0)))
        self._pub_command.publish(msg)

    def _publish_end_effector_pose(self):
        tcp = self._kin.fk(self._kin.q_from_joints(self._joints), frame="tcp")
        x, y, z, w = _quat(tcp.rotation)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = tcp.translation.tolist()
        msg.pose.orientation.x, msg.pose.orientation.y = x, y
        msg.pose.orientation.z, msg.pose.orientation.w = z, w
        self._pub_ee_pose.publish(msg)


def _quat(R):
    import pinocchio
    q = pinocchio.Quaternion(R)
    return float(q.x), float(q.y), float(q.z), float(q.w)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
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
