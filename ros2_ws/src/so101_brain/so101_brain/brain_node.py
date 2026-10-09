#!/usr/bin/env python3
"""brain_node.py - intelligence of the arm: kinematics, then pick & place.

Never talks to the backend: reads /joint_states, commands through /joint_command.

  - Forward kinematics: from /joint_states, publish the TCP pose
    (/end_effector_pose, geometry_msgs/PoseStamped) to show in RViz
  - Inverse kinematics: reach an XYZ target (position-only, damped least squares)

TODO:
  - Later: pick & place state machine
"""
from pathlib import Path

import numpy as np
import pinocchio
import rclpy
import rclpy.node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from so101_interfaces.srv import GoToTarget

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
JOINT_NAMES = ARM_JOINT_NAMES + ["gripper"]

BASE_FRAME = "base_link"
TCP_FRAME = "gripper_frame_link"
# world -> base_link is the identity (robot base at the MuJoCo origin, see viz.launch.py).
ACCEPTED_TARGET_FRAMES = ("", BASE_FRAME, "world")

IK_MAX_ITERS = 500
IK_TOLERANCE = 1e-3  # m
IK_DAMPING = 1e-2
IK_MAX_STEP = 0.2  # rad per iteration, keeps the solver from jumping around
IK_RESTARTS = 10  # random seeds tried when starting from the current pose fails


class BrainNode(rclpy.node.Node):
    """Brain of the arm: kinematics, then pick & place."""

    def __init__(self):
        super().__init__("so101_brain")

        urdf = Path(get_package_share_directory("so101_description")) / "urdf" / "so101.urdf"
        self._model = pinocchio.buildModelFromUrdf(str(urdf))
        self._data = self._model.createData()
        self._tcp_frame_id = self._model.getFrameId(TCP_FRAME)

        # Joint name -> index in q / column in the Jacobian (pinocchio has its own order).
        self._idx_q = {n: self._model.joints[self._model.getJointId(n)].idx_q for n in JOINT_NAMES}
        self._idx_v = {n: self._model.joints[self._model.getJointId(n)].idx_v for n in JOINT_NAMES}
        self._arm_idx_v = [self._idx_v[n] for n in ARM_JOINT_NAMES]

        self._q = None  # latest configuration from /joint_states
        self._rng = np.random.default_rng()

        self._pub_ee_pose = self.create_publisher(PoseStamped, "/end_effector_pose", 10)
        self._pub_command = self.create_publisher(JointState, "/joint_command", 10)
        self.create_subscription(JointState, "/joint_states", self._cb_joint_states, 10)
        self.create_service(GoToTarget, "/go_to_target", self._cb_go_to_target)

        self.get_logger().info("Brain node ready.")

    def _cb_go_to_target(self, request, response):
        """Move the TCP to request.target_pose position (orientation is ignored)."""
        frame = request.target_pose.header.frame_id
        if frame not in ACCEPTED_TARGET_FRAMES:
            response.success = False
            response.message = f"Unsupported frame '{frame}', use '{BASE_FRAME}'"
            return response

        p = request.target_pose.pose.position
        q = self._solve_ik([p.x, p.y, p.z])
        if q is None:
            response.success = False
            response.message = f"Target ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) unreachable"
            return response

        self._send_joint_command(q)
        response.success = True
        response.message = "Joint command sent: " + ", ".join(
            f"{n}={q[self._idx_q[n]]:.3f}" for n in ARM_JOINT_NAMES
        )
        return response

    def _cb_joint_states(self, msg: JointState):
        q = self._q.copy() if self._q is not None else pinocchio.neutral(self._model)
        for name, pos in zip(msg.name, msg.position):
            if name in self._idx_q:
                q[self._idx_q[name]] = pos
        self._q = q
        self._publish_end_effector_pose()

    def _forward_kinematics(self, q):
        """Return the TCP placement (pinocchio.SE3) in the base frame."""
        pinocchio.forwardKinematics(self._model, self._data, q)
        pinocchio.updateFramePlacements(self._model, self._data)
        return self._data.oMf[self._tcp_frame_id]

    def _publish_end_effector_pose(self):
        if self._q is None:
            return
        tcp = self._forward_kinematics(self._q)
        quat = pinocchio.Quaternion(tcp.rotation)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = tcp.translation.tolist()
        msg.pose.orientation.x = quat.x
        msg.pose.orientation.y = quat.y
        msg.pose.orientation.z = quat.z
        msg.pose.orientation.w = quat.w
        self._pub_ee_pose.publish(msg)

    def _solve_ik(self, target_position):
        """Position-only IK: joint configuration putting the TCP at target_position.

        Starts from the current configuration, then retries from random seeds if
        stuck in a local minimum. Returns q, or None if not reached.
        """
        target = np.asarray(target_position, dtype=float)
        q0 = self._q.copy() if self._q is not None else pinocchio.neutral(self._model)
        lower, upper = self._model.lowerPositionLimit, self._model.upperPositionLimit

        best_err = np.inf
        for attempt in range(IK_RESTARTS + 1):
            seed = q0 if attempt == 0 else self._rng.uniform(lower, upper)
            seed[self._idx_q["gripper"]] = q0[self._idx_q["gripper"]]
            q, err = self._solve_ik_from(target, seed)
            if err < IK_TOLERANCE:
                return q
            best_err = min(best_err, err)

        self.get_logger().warn(
            f"IK did not converge for {target.tolist()} (residual {best_err * 1000:.1f} mm)"
        )
        return None

    def _solve_ik_from(self, target, q):
        """Damped least squares from seed q. Returns (q, position error norm)."""
        lower, upper = self._model.lowerPositionLimit, self._model.upperPositionLimit

        for _ in range(IK_MAX_ITERS):
            err = target - self._forward_kinematics(q).translation
            if np.linalg.norm(err) < IK_TOLERANCE:
                break

            # Linear part of the TCP Jacobian, arm joints only (the gripper does not move the TCP).
            jac = pinocchio.computeFrameJacobian(
                self._model, self._data, q, self._tcp_frame_id, pinocchio.LOCAL_WORLD_ALIGNED
            )[:3, self._arm_idx_v]
            # Damped least squares: dq = J^T (J J^T + l^2 I)^-1 err
            dq_arm = jac.T @ np.linalg.solve(jac @ jac.T + IK_DAMPING**2 * np.eye(3), err)
            dq_arm = np.clip(dq_arm, -IK_MAX_STEP, IK_MAX_STEP)

            dq = np.zeros(self._model.nv)
            dq[self._arm_idx_v] = dq_arm
            q = np.clip(pinocchio.integrate(self._model, q, dq), lower, upper)

        return q, np.linalg.norm(target - self._forward_kinematics(q).translation)

    def _send_joint_command(self, q, gripper_pct=None):
        """Send arm joints (rad) and optionally the gripper (0-100 %) to the driver."""
        q = np.clip(q, self._model.lowerPositionLimit, self._model.upperPositionLimit)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(ARM_JOINT_NAMES)
        msg.position = [float(q[self._idx_q[n]]) for n in ARM_JOINT_NAMES]
        if gripper_pct is not None:
            msg.name.append("gripper")
            msg.position.append(float(np.clip(gripper_pct, 0.0, 100.0)))
        self._pub_command.publish(msg)


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
