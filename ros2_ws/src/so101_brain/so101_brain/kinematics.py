"""kinematics.py - forward / inverse kinematics of the SO-ARM101 (pinocchio).

Pure Python (no ROS): used by brain_node and by tools/eval_pick_place.py.

Frames (URDF, checked against the MuJoCo model):
  - base_link: robot base, identical to the MuJoCo world frame.
  - gripper_frame_link ("tcp"): sits on the tip of the FIXED jaw. Local +z is
    the approach axis (out of the jaws), the moving jaw closes along local -x.
Frames added here, rigidly attached to the TCP:
  - "grasp": where the centre of the cube must be when the jaws close. It is
    6 mm under the fixed jaw tip: the fixed jaw goes straight down into the
    MIDDLE of the vertical diamond tunnel (34 mm diagonal, so about ±13 mm of
    clearance around the 7 mm finger), 19 mm deep, while the moving jaw stays
    outside the cube.
  - "held": where the centre of the cube ends up once grasped. Closing pushes the
    cube until the diamond vertex hits the fixed jaw: the V self-centres the cube
    on the finger (form closure). Measured in MuJoCo: the same held pose to
    ±1 mm for errors of ±10 mm in x and y and ±15 deg in yaw at the grasp.
    It is the frame driven to the box for the drop.
Offsets measured in MuJoCo (tools/eval_pick_place.py and grasp sweeps).

IK: damped least squares, position only, position + orientation, or position
first with the orientation in its null space ("soft"). Joint limits are handled
with an active set. For pick & place the gripper points straight down with a
yaw around the vertical: with 5 DoF this is exactly reachable for low targets
(pan + 3 pitch joints set x, y, z and the pitch, wrist_roll sets the yaw).
"""
from dataclasses import dataclass

import numpy as np
import pinocchio

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"
JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_JOINT]
TCP_FRAME = "gripper_frame_link"

# In the TCP frame (m).
DEFAULT_GRASP_OFFSET = (0.0, 0.0, 0.006)
DEFAULT_HELD_OFFSET = (0.019, 0.0, -0.006)


def top_down_rotation(yaw: float) -> np.ndarray:
    """TCP rotation pointing straight down, jaw axis (local x) at `yaw` in the table plane."""
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


@dataclass
class IKResult:
    q: np.ndarray            # full pinocchio configuration
    position_error: float    # m
    rotation_error: float    # rad (0 if no orientation target)

    def ok(self, pos_tol: float, rot_tol: float = np.inf) -> bool:
        return self.position_error < pos_tol and self.rotation_error < rot_tol


class ArmKinematics:
    """FK/IK of the SO-ARM101 from its URDF."""

    def __init__(self, urdf_path: str, grasp_offset=DEFAULT_GRASP_OFFSET,
                 held_offset=DEFAULT_HELD_OFFSET, damping: float = 1e-2, max_step: float = 0.2,
                 rotation_weight: float = 0.3, seed: int | None = None):
        self.model = pinocchio.buildModelFromUrdf(str(urdf_path))
        tcp_id = self.model.getFrameId(TCP_FRAME)
        self.frames = {
            "tcp": tcp_id,
            "grasp": self._add_tcp_frame("grasp_point", tcp_id, grasp_offset),
            "held": self._add_tcp_frame("held_object", tcp_id, held_offset),
        }
        self.data = self.model.createData()
        self.idx_q = {n: self.model.joints[self.model.getJointId(n)].idx_q for n in JOINT_NAMES}
        self.idx_v = {n: self.model.joints[self.model.getJointId(n)].idx_v for n in JOINT_NAMES}
        self.arm_idx_q = [self.idx_q[n] for n in ARM_JOINT_NAMES]
        self.arm_idx_v = [self.idx_v[n] for n in ARM_JOINT_NAMES]
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()
        self.damping = damping
        self.max_step = max_step
        self.rotation_weight = rotation_weight  # m per rad: trades position vs orientation error
        self._rng = np.random.default_rng(seed)

    def _add_tcp_frame(self, name, tcp_id, offset):
        tcp = self.model.frames[tcp_id]
        parent_joint = getattr(tcp, "parentJoint", None)
        if parent_joint is None:  # pinocchio 2
            parent_joint = tcp.parent
        placement = tcp.placement * pinocchio.SE3(np.eye(3), np.asarray(offset, dtype=float))
        return self.model.addFrame(
            pinocchio.Frame(name, parent_joint, tcp_id, placement, pinocchio.FrameType.OP_FRAME)
        )

    # ------------------------------------------------------------ helpers

    def neutral(self) -> np.ndarray:
        return pinocchio.neutral(self.model)

    def q_from_joints(self, joints: dict[str, float], base: np.ndarray | None = None) -> np.ndarray:
        q = self.neutral() if base is None else base.copy()
        for name, value in joints.items():
            if name in self.idx_q:
                q[self.idx_q[name]] = value
        return q

    def joints_from_q(self, q: np.ndarray, names=ARM_JOINT_NAMES) -> dict[str, float]:
        return {n: float(q[self.idx_q[n]]) for n in names}

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower, self.upper)

    def fk(self, q: np.ndarray, frame: str = "grasp") -> pinocchio.SE3:
        """Placement of `frame` ("tcp", "grasp" or "held") in base_link."""
        pinocchio.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.frames[frame]]

    # ----------------------------------------------------------------- IK

    def solve(self, position, q_seed: np.ndarray, rotation: np.ndarray | None = None,
              frame: str = "grasp", soft_rotation: bool = False, max_iters: int = 200,
              tol: float = 1e-4) -> IKResult:
        """Damped least squares from one seed.

        soft_rotation=False: position and orientation are one task (both must be reachable).
        soft_rotation=True: task priority, the position is reached first and the
        orientation is only tracked in the null space of the position task
        (as close as possible when it cannot be reached exactly).
        """
        target = np.asarray(position, dtype=float)
        frame_id = self.frames[frame]
        q = self.clamp(q_seed.copy())
        arm_q = self.arm_idx_q
        for _ in range(max_iters):
            pose = self.fk(q, frame)
            err_p = target - pose.translation
            err_r = None if rotation is None else pinocchio.log3(rotation @ pose.rotation.T)
            if np.linalg.norm(err_p) < tol and (err_r is None or np.linalg.norm(err_r) < tol):
                break
            jac = pinocchio.computeFrameJacobian(
                self.model, self.data, q, frame_id, pinocchio.LOCAL_WORLD_ALIGNED
            )[:, self.arm_idx_v]
            # Active set: a joint sitting on a limit and pushed outward is locked (its
            # Jacobian column removed) so the other joints compensate, instead of the
            # clamp silently breaking the task.
            free = np.ones(len(arm_q), dtype=bool)
            for _ in range(len(arm_q)):
                dq_arm = self._step(jac * free, err_p, err_r, soft_rotation) * free
                blocked = ((q[arm_q] <= self.lower[arm_q] + 1e-6) & (dq_arm < 0)) | \
                          ((q[arm_q] >= self.upper[arm_q] - 1e-6) & (dq_arm > 0))
                if not np.any(blocked & free):
                    break
                free &= ~blocked
            if np.linalg.norm(dq_arm) < 1e-7:  # converged as far as it can
                break
            dq = np.zeros(self.model.nv)
            dq[self.arm_idx_v] = np.clip(dq_arm, -self.max_step, self.max_step)
            q = self.clamp(pinocchio.integrate(self.model, q, dq))
        if soft_rotation and rotation is not None:
            # The damped null-space projection leaves a few mm of position residual:
            # polish the position alone (tiny joint motion, orientation nearly unchanged).
            if np.linalg.norm(target - self.fk(q, frame).translation) > tol:
                q = self.solve(target, q, None, frame, max_iters=50, tol=tol).q
        return self._result(q, target, rotation, frame)

    def _step(self, jac, err_p, err_r, soft_rotation):
        """One damped least squares step: dq = J^T (J J^T + l^2 I)^-1 err."""
        damp = self.damping**2
        jac_p = jac[:3]
        if err_r is None:
            return jac_p.T @ np.linalg.solve(jac_p @ jac_p.T + damp * np.eye(3), err_p)
        if soft_rotation:  # orientation in the null space of the position task
            jp_pinv = jac_p.T @ np.linalg.inv(jac_p @ jac_p.T + damp * np.eye(3))
            dq_p = jp_pinv @ err_p
            null = np.eye(jac.shape[1]) - jp_pinv @ jac_p
            jr = jac[3:] @ null
            dq_r = jr.T @ np.linalg.solve(jr @ jr.T + damp * np.eye(3), err_r - jac[3:] @ dq_p)
            return dq_p + null @ dq_r
        w = self.rotation_weight
        err = np.concatenate([err_p, w * err_r])
        jac_s = np.vstack([jac_p, w * jac[3:]])
        return jac_s.T @ np.linalg.solve(jac_s @ jac_s.T + damp * np.eye(6), err)

    def _random_seed(self, q_ref: np.ndarray) -> np.ndarray:
        seed = self._rng.uniform(self.lower, self.upper)
        seed[self.idx_q[GRIPPER_JOINT]] = q_ref[self.idx_q[GRIPPER_JOINT]]
        return seed

    def solve_global(self, position, q_seed: np.ndarray, rotation: np.ndarray | None = None,
                     frame: str = "grasp", restarts: int = 8, pos_tol: float = 1e-3,
                     rot_tol: float = np.radians(3)) -> IKResult:
        """Seed first, then random restarts; returns the best solution found."""
        best = None
        for attempt in range(restarts + 1):
            seed = q_seed if attempt == 0 else self._random_seed(q_seed)
            res = self.solve(position, seed, rotation, frame)
            if best is None or self._cost(res) < self._cost(best):
                best = res
            if best.ok(pos_tol, rot_tol):
                break
        return best

    def solve_top_down(self, position, q_seed: np.ndarray, yaw: float, frame: str = "grasp",
                       symmetry: float = np.pi / 2, pos_tol: float = 1e-3,
                       rot_tol: float = np.radians(3)) -> tuple[IKResult, float]:
        """Gripper pointing down at `position`. The object is symmetric by `symmetry`
        around z, so every yaw + k*symmetry is valid: keep the reachable one that
        moves the joints the least from q_seed. Falls back to an exact position with
        the least possible tilt. Returns (result, yaw used)."""
        n = max(1, int(round(2 * np.pi / symmetry)))
        yaws = [yaw + k * symmetry for k in range(n)]
        best, best_yaw, best_cost = None, yaw, np.inf
        for y in yaws:
            res = self.solve_global(position, q_seed, top_down_rotation(y), frame, restarts=4,
                                    pos_tol=pos_tol, rot_tol=rot_tol)
            if res.ok(pos_tol, rot_tol):
                cost = self.joint_distance(res.q, q_seed)
                if cost < best_cost:
                    best, best_yaw, best_cost = res, y, cost
        if best is not None:
            return best, best_yaw
        for y in yaws:
            for attempt in range(3):
                seed = q_seed if attempt == 0 else self._random_seed(q_seed)
                res = self.solve(position, seed, top_down_rotation(y), frame, soft_rotation=True)
                if res.position_error < pos_tol and (best is None or res.rotation_error < best.rotation_error):
                    best, best_yaw = res, y
        if best is None:  # unreachable: best effort, for diagnosis
            best = self.solve_global(position, q_seed, top_down_rotation(yaw), frame, restarts=4)
        return best, best_yaw

    def joint_distance(self, q_a: np.ndarray, q_b: np.ndarray) -> float:
        return float(np.linalg.norm((q_a - q_b)[self.arm_idx_q]))

    def _result(self, q, target, rotation, frame) -> IKResult:
        pose = self.fk(q, frame)
        rot_err = 0.0 if rotation is None else float(np.linalg.norm(pinocchio.log3(rotation @ pose.rotation.T)))
        return IKResult(q, float(np.linalg.norm(target - pose.translation)), rot_err)

    @staticmethod
    def _cost(res: IKResult) -> float:
        return res.position_error + 0.01 * res.rotation_error
