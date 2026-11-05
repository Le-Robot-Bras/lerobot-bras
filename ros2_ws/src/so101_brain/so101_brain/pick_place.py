"""pick_place.py - pick & place state machine of the SO-ARM101 (pure Python, no ROS).

brain_node feeds it joint states and perception, calls step() at a fixed rate
and publishes the returned command on /joint_command. tools/eval_pick_place.py
drives the very same class directly against the MuJoCo simulation.

    IDLE -> HOME -> OBSERVE -> APPROACH -> DESCEND -> GRASP -> LIFT -> TRANSPORT -> RELEASE -> RETREAT -> DONE
                       ^                                 |        |
                       +--------- RETRY (grasp failed) <-+--------+

Robustness (see the bootstrap slides):
  - Perception never blocks control: the latest estimate is used, and the arm only
    commits to a target in OBSERVE, from the folded HOME pose where the camera sees
    the whole table (the arm occludes the cube when it is above it).
  - Object moved: during APPROACH a fresh, consistent estimate far from the target re-plans.
  - Grasp failed: detected by the gripper opening (a good grasp closes fully: the
    fixed jaw is inside the middle of the cube tunnel, see kinematics.py) and by perception (cube still on the table
    after LIFT). Retried with a small lateral offset, up to max_attempts.
  - Object lost / target unreachable: back to HOME and wait (never moves blindly).
"""
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .kinematics import ARM_JOINT_NAMES, ArmKinematics, top_down_rotation


class State(str, Enum):
    IDLE = "IDLE"
    HOME = "HOME"
    OBSERVE = "OBSERVE"
    APPROACH = "APPROACH"
    DESCEND = "DESCEND"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    RETRY = "RETRY"
    DONE = "DONE"


@dataclass
class Observation:
    position: np.ndarray   # object centre, robot frame (m)
    yaw: float             # rad (cube: modulo pi/2)
    stamp: float           # s


@dataclass
class Command:
    joints: dict           # arm joint name -> rad
    gripper: float         # 0 (closed) - 100 (open) %


@dataclass
class PickPlaceConfig:
    # Arm straight up ("candle"): hides the least of the table from the external
    # camera (searched with MuJoCo segmentation renders), rad.
    home: dict = field(default_factory=lambda: {
        "shoulder_pan": 0.0, "shoulder_lift": -0.6, "elbow_flex": -1.0,
        "wrist_flex": 0.0, "wrist_roll": 0.0})
    gripper_open: float = 45.0          # % : jaws 6 cm apart, clears the 5 cm cube
    gripper_closed: float = 0.0
    grasp_hold_max: float = 10.0        # % : closed wider than this = bad grasp
    approach_height: float = 0.05       # m above the grasp point
    lift_height: float = 0.10           # m, grasp point after lift
    drop_height: float = 0.09           # m, held cube centre above the box (walls: 5 cm)
    joint_speed: float = 1.0            # rad/s, joint-space moves
    cart_speed: float = 0.05            # m/s, straight-line moves
    descend_speed: float = 0.03         # m/s
    gripper_speed: float = 100.0        # %/s
    settle_time: float = 0.4            # s, wait at the end of each move
    reach_tolerance: float = 0.05       # rad, measured joints vs. command at the end of a move
    move_timeout: float = 3.0           # s after a move before giving up waiting for the arm
    observe_samples: int = 5            # consistent detections needed to commit
    observe_spread: float = 0.004       # m, max spread of those detections
    observation_timeout: float = 1.0    # s, an estimate older than this is ignored
    replan_distance: float = 0.02       # m, object moved during APPROACH
    grasp_check_distance: float = 0.02  # m, cube still there after LIFT = grasp failed
    max_attempts: int = 4
    # Adaptive lateral search: when a grasp is not hooked, the closing jaws push the cube
    # sideways by ~1 cm, AWAY from the side the jaws were offset to. Perception still sees
    # the cube at that moment (score ~0.8): the sign of that shift tells which way to
    # correct the next attempt. Step halves when the sign flips (bisection).
    adaptive_retry: bool = False        # old edge grasp, 80 seeds: 75-76/80 vs 77/80 with the fixed offsets
    retry_step: float = 0.003           # m, first lateral correction
    retry_min_shift: float = 0.004      # m, perceived shift needed to trust the sign
    retry_halve: bool = True            # halve the step when the sign flips
    retry_unseen_flip: bool = True      # cube not seen after a failure = overshoot
    retry_rotate: bool = False          # retries grasp the other pair of faces (jaws +90 deg); measured worse in sim
    # m, lateral (along the jaws) offset per attempt. Safety net only: since the fixed jaw
    # enters the middle of the tunnel (±10 mm tolerance), seeds 0-199 all succeed at the
    # first attempt (the old grasp near the edge of the tunnel needed them: 77/80).
    retry_offsets: tuple = (-0.001, 0.001, -0.003, 0.003)


class Trajectory:
    """Smooth (minimum-jerk) interpolation between two joint vectors or two points."""

    def __init__(self, start, goal, duration, t0):
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.duration = max(duration, 1e-3)
        self.t0 = t0

    def sample(self, now):
        s = min(max((now - self.t0) / self.duration, 0.0), 1.0)
        s = s * s * s * (10 - 15 * s + 6 * s * s)
        return self.start + (self.goal - self.start) * s

    def finished(self, now):
        return now - self.t0 >= self.duration


class PickPlace:
    def __init__(self, kin: ArmKinematics, config: PickPlaceConfig | None = None, log=None):
        self.kin = kin
        self.cfg = config or PickPlaceConfig()
        self.log = log or (lambda level, msg: None)
        self.state = State.IDLE
        self.status = "waiting for /brain/start"
        self.attempt = 0
        self.successes = 0

        self._q_meas = None            # measured configuration (pinocchio vector)
        self._gripper_meas = None      # measured gripper %
        self._q_cmd = None             # last commanded configuration
        self._gripper_cmd = self.cfg.gripper_open
        self._object = None            # latest Observation
        self._box = None
        self._object_history = deque(maxlen=self.cfg.observe_samples)

        self._target = None            # committed object position / yaw / jaw yaw
        self._target_yaw = 0.0
        self._jaw_yaw = 0.0
        self._drop = None
        self._box_yaw = 0.0
        self._reset_search()
        self._unreachable = None       # last target refused, to avoid re-planning it in a loop

        # current motion
        self._mode = None              # "joint" | "line" | "gripper" | "wait"
        self._traj = None
        self._line_rot = None
        self._gripper_traj = None
        self._move_end = None
        self._state_t0 = 0.0

    # ------------------------------------------------------------------ inputs

    def update_joints(self, joints: dict, gripper_pct: float):
        self._q_meas = self.kin.q_from_joints(joints, self._q_meas)
        self._gripper_meas = gripper_pct

    def update_object(self, obs: Observation):
        self._object = obs
        self._object_history.append(obs)

    def update_box(self, obs: Observation):
        self._box = obs

    # ---------------------------------------------------------------- services

    def start(self, now) -> tuple[bool, str]:
        if self._q_meas is None:
            return False, "no /joint_states yet"
        if self.state not in (State.IDLE, State.DONE):
            return False, f"already running ({self.state.value})"
        self.attempt = 0
        self._reset_search()
        # Restart from DONE: drop any motion still running (e.g. the final move home).
        self._mode = None
        self._traj = None
        self._gripper_traj = None
        self._unreachable = None
        self._q_cmd = self._q_meas.copy()
        self._gripper_cmd = self._gripper_meas if self._gripper_meas is not None else self.cfg.gripper_open
        self._enter(State.HOME, now, "going home")
        return True, "pick & place started"

    def stop(self, now) -> tuple[bool, str]:
        """Freeze where the arm is (it keeps the last command, holding the cube if any)."""
        was = self.state
        self._mode = None
        if self._q_meas is not None:
            self._q_cmd = self._q_meas.copy()
        self._enter(State.IDLE, now, "stopped")
        return True, f"stopped (was {was.value})"

    # ------------------------------------------------------------ visualization

    HELD_STATES = (State.LIFT, State.TRANSPORT)
    PLACED_STATES = (State.RELEASE, State.RETREAT, State.DONE)

    def belief(self) -> tuple[str, np.ndarray | None]:
        """Where the brain believes the cube is (for RViz only, no effect on behaviour):
        ('table', perceived position), ('held', centre in the gripper), ('placed', in the box)
        or ('unknown', None)."""
        if self.state in self.HELD_STATES and self._q_cmd is not None:
            return "held", self.kin.fk(self._q_cmd, "held").translation.copy()
        if self.state in self.PLACED_STATES and self._drop is not None:
            p = self._drop.copy()
            p[2] = self._object.position[2] if self._object is not None else 0.025
            return "placed", p
        if self._object is not None:
            return "table", np.asarray(self._object.position, dtype=float).copy()
        return "unknown", None

    def cube_yaw(self) -> float:
        """Yaw to draw the believed cube with."""
        if self.state in self.HELD_STATES + self.PLACED_STATES:
            return float(self._jaw_yaw)
        return float(self._object.yaw) if self._object is not None else 0.0

    def targets(self) -> dict:
        """Current planned points: pre-grasp (above the cube) and drop point, when known."""
        out = {}
        if self._target is not None and self.state not in (State.IDLE, State.HOME, State.OBSERVE, State.DONE):
            out["pregrasp"] = self._target + [0.0, 0.0, self.cfg.approach_height]
            out["grasp"] = self._target.copy()
        if self._drop is not None and self.state not in (State.IDLE, State.DONE):
            out["drop"] = self._drop.copy()
        return out

    # -------------------------------------------------------------------- loop

    def step(self, now) -> Command | None:
        if self.state == State.IDLE or self._q_meas is None or self._q_cmd is None:
            return None
        if self._mode is not None:
            if not self._advance_motion(now):
                return self._command()
            self._mode = None
        self._on_motion_done(now)
        return self._command()

    # ------------------------------------------------------------ state logic

    def _on_motion_done(self, now):
        cfg = self.cfg
        s = self.state

        if s == State.HOME:
            if self._q_cmd is not None and self._distance_to_home() > 1e-3:
                self._move_joints(self.kin.q_from_joints(cfg.home, self._q_cmd), now)
                self._set_gripper(cfg.gripper_open, now, parallel=True)
                return
            self._object_history.clear()
            self._enter(State.OBSERVE, now, "looking for the cube and the box")

        elif s == State.OBSERVE:
            target = self._stable_object(now)
            box = self._fresh(self._box, now, timeout=math.inf)
            if target is None or box is None:
                self.status = "waiting for a stable detection of the cube and the box"
                return
            if self._plan_pick(target, box, now):
                self._enter(State.APPROACH, now, f"attempt {self.attempt + 1}: approaching the cube")

        elif s == State.APPROACH:
            # The pre-grasp pose has been reached (joint move). Last chance to re-plan.
            fresh = self._fresh(self._object, now)
            if fresh is not None and np.linalg.norm((fresh.position - self._target)[:2]) > cfg.replan_distance:
                self.log("warn", "cube moved during approach: re-planning")
                self._enter(State.HOME, now, "cube moved, re-observing")
                return
            self._move_line(self._target, self._jaw_yaw, cfg.descend_speed, now)
            self._enter(State.DESCEND, now, "descending")

        elif s == State.DESCEND:
            self._set_gripper(cfg.gripper_closed, now)
            self._enter(State.GRASP, now, "closing the gripper")

        elif s == State.GRASP:
            if self._gripper_meas is not None and self._gripper_meas > cfg.grasp_hold_max:
                self._grasp_failed(now, f"gripper stopped at {self._gripper_meas:.0f}% (fixed jaw not in the tunnel)",
                                   learn=True)
                return
            up = np.array([self._target[0], self._target[1], cfg.lift_height])
            self._move_line(up, self._jaw_yaw, cfg.cart_speed, now)
            self._enter(State.LIFT, now, "lifting")

        elif s == State.LIFT:
            seen = self._fresh(self._object, now, since=self._state_t0 + 0.3)
            if seen is not None and np.linalg.norm((seen.position - self._target)[:2]) < cfg.grasp_check_distance:
                self._grasp_failed(now, "cube still on the table after lift")
                return
            if self._gripper_meas is not None and self._gripper_meas > cfg.grasp_hold_max:
                self._grasp_failed(now, "gripper opened during lift")
                return
            # Jaw axis aligned with a box wall: the in-hand uncertainty of the cube is along
            # the jaws, so it stays along a box axis where the clearance is largest.
            res, yaw = self.kin.solve_top_down(self._drop, self._q_cmd, self._box_yaw, frame="held",
                                               symmetry=np.pi / 2)
            if res.position_error > 0.002:  # not reachable box-aligned: any yaw
                res, yaw = self.kin.solve_top_down(self._drop, self._q_cmd, self._jaw_yaw, frame="held",
                                                   symmetry=np.pi / 12)
            if res.position_error > 0.005:
                self._abort(now, f"drop point unreachable ({res.position_error * 1000:.0f} mm)")
                return
            self._move_joints(res.q, now)
            self._enter(State.TRANSPORT, now, "transporting to the box")

        elif s == State.TRANSPORT:
            self._set_gripper(cfg.gripper_open, now)
            self._enter(State.RELEASE, now, "releasing")

        elif s == State.RELEASE:
            up = self.kin.fk(self._q_cmd).translation + [0.0, 0.0, 0.03]
            self._move_line(up, None, cfg.cart_speed, now)
            self._enter(State.RETREAT, now, "retreating")

        elif s == State.RETREAT:
            self.successes += 1
            self.log("info", f"cube dropped in the box (attempt {self.attempt + 1})")
            self._move_joints(self.kin.q_from_joints(cfg.home, self._q_cmd), now)
            self._enter(State.DONE, now, "done: cube in the box")

        elif s == State.RETRY:
            # Gripper is open: back up vertically before going home (no dragging on the table).
            here = self.kin.fk(self._q_cmd).translation
            if here[2] < self._target[2] + cfg.approach_height - 0.005:
                self._move_line(here + [0.0, 0.0, cfg.approach_height], self._jaw_yaw, cfg.cart_speed, now)
                return
            self._enter(State.HOME, now, "retrying from home")

        elif s == State.DONE:
            self.status = f"done ({self.successes} cube(s) placed), call /brain/start again"

    # --------------------------------------------------------------- planning

    def _plan_pick(self, target: Observation, box: Observation, now) -> bool:
        cfg = self.cfg
        offset = 0.0 if cfg.adaptive_retry else cfg.retry_offsets[self.attempt % len(cfg.retry_offsets)]
        # The grasp itself must be exactly top-down; the pre-grasp above it may tilt a little.
        grasp, jaw_yaw = None, target.yaw
        if self.attempt > 0 and cfg.retry_rotate:
            # Retry on the other pair of faces: the jaws turned by 90 deg see the
            # perception error along another axis (a failed hook is usually lateral).
            grasp, jaw_yaw = self.kin.solve_top_down(target.position, self._q_cmd,
                                                     self._jaw_yaw + np.pi / 2 * self.attempt,
                                                     symmetry=2 * np.pi)
            if not grasp.ok(2e-3, np.radians(5)):
                grasp = None
        if grasp is None:
            grasp, jaw_yaw = self.kin.solve_top_down(target.position, self._q_cmd, target.yaw)
        if not grasp.ok(2e-3, np.radians(5)):
            if self._unreachable is None or np.linalg.norm(self._unreachable - target.position) > 0.01:
                self._unreachable = target.position.copy()
                self.log("error", f"cube at {np.round(target.position, 3).tolist()} unreachable top-down "
                                  f"({grasp.position_error * 1000:.0f} mm, {np.degrees(grasp.rotation_error):.0f} deg)")
            self.status = "cube out of reach, waiting for it to be moved"
            return False
        self._unreachable = None
        jaw_y = np.array([-np.sin(jaw_yaw), np.cos(jaw_yaw), 0.0])
        self._target = target.position + offset * jaw_y
        if cfg.adaptive_retry:
            self._target = self._target + self._lat_corr
        self._target_yaw = target.yaw
        self._jaw_yaw = jaw_yaw
        self._drop = np.array([box.position[0], box.position[1], cfg.drop_height])
        self._box_yaw = box.yaw
        pre = self._target + [0.0, 0.0, cfg.approach_height]
        res = self.kin.solve(pre, grasp.q, top_down_rotation(jaw_yaw), soft_rotation=True)
        if res.position_error > 2e-3:
            self.log("error", f"pre-grasp unreachable ({res.position_error * 1000:.0f} mm)")
            self.status = "pre-grasp out of reach"
            return False
        self._move_joints(res.q, now)
        self.log("info", f"target cube {np.round(self._target, 3).tolist()} yaw {np.degrees(jaw_yaw):.0f} deg, "
                         f"box {np.round(self._drop[:2], 3).tolist()}")
        return True

    def _learn_from_failure(self, now):
        """Adaptive lateral search (see PickPlaceConfig.adaptive_retry)."""
        cfg = self.cfg
        if self._q_meas is None:
            return
        seen = self._fresh(self._object, now, since=self._state_t0)
        grasp = self.kin.fk(self._q_meas)
        jaw_y = grasp.rotation[:, 1].copy()
        jaw_y[2] = 0.0
        jaw_y /= np.linalg.norm(jaw_y)
        shift = None if seen is None else float((seen.position - grasp.translation)[:2] @ jaw_y[:2])
        if shift is not None and abs(shift) >= cfg.retry_min_shift:
            sign = np.sign(shift)
        elif self._last_sign is not None and cfg.retry_unseen_flip:
            # Not seen (pushed under the gripper, low fit score): measured in sim, this
            # happens when the cube is pushed the other way, i.e. we overshot.
            sign = -self._last_sign
            shift = 0.0
        else:
            return
        if self._last_sign is not None and sign != self._last_sign and cfg.retry_halve:
            self._lat_step /= 2
        self._last_sign = sign
        self._lat_corr = self._lat_corr + sign * self._lat_step * jaw_y
        self.log("info", f"cube pushed {shift * 1000:+.0f} mm sideways: lateral correction now "
                         f"{np.round(self._lat_corr[:2] * 1000, 1).tolist()} mm")

    def _grasp_failed(self, now, why, learn=False):
        if learn and self.cfg.adaptive_retry:
            self._learn_from_failure(now)
        self.attempt += 1
        if self.attempt >= self.cfg.max_attempts:
            self._abort(now, f"grasp failed {self.attempt} times ({why})")
            self.attempt = 0
            return
        self.log("warn", f"grasp failed: {why}. Retrying ({self.attempt + 1}/{self.cfg.max_attempts})")
        # Open, back up vertically (RETRY), then re-observe from home.
        self._set_gripper(self.cfg.gripper_open, now)
        self._enter(State.RETRY, now, f"grasp failed ({why}), retrying")

    def _reset_search(self):
        self._lat_corr = np.zeros(3)
        self._lat_step = self.cfg.retry_step
        self._last_sign = None

    def _abort(self, now, why):
        self.log("error", why)
        self._enter(State.HOME, now, why)

    def _stable_object(self, now):
        cfg = self.cfg
        recent = [o for o in self._object_history if now - o.stamp < cfg.observation_timeout]
        if len(recent) < cfg.observe_samples:
            return None
        pos = np.array([o.position for o in recent])
        if np.max(np.linalg.norm(pos - np.median(pos, axis=0), axis=1)) > cfg.observe_spread:
            return None
        yaws = np.array([o.yaw for o in recent])
        # circular mean modulo pi/2
        yaw = math.atan2(np.mean(np.sin(4 * yaws)), np.mean(np.cos(4 * yaws))) / 4
        return Observation(np.median(pos, axis=0), yaw, recent[-1].stamp)

    def _fresh(self, obs, now, timeout=None, since=None):
        if obs is None:
            return None
        if now - obs.stamp > (self.cfg.observation_timeout if timeout is None else timeout):
            return None
        if since is not None and obs.stamp < since:
            return None
        return obs

    def _distance_to_home(self):
        home = self.kin.q_from_joints(self.cfg.home, self._q_cmd)
        return self.kin.joint_distance(home, self._q_cmd)

    # ---------------------------------------------------------------- motion

    def _enter(self, state, now, status):
        if state != self.state:
            self.log("info", f"{self.state.value} -> {state.value}: {status}")
        self.state = state
        self.status = status
        self._state_t0 = now

    def _move_joints(self, q_goal, now):
        q_goal = self.kin.clamp(q_goal)
        dist = np.max(np.abs((q_goal - self._q_cmd)[self.kin.arm_idx_q]))
        self._traj = Trajectory(self._q_cmd, q_goal, 1.875 * dist / self.cfg.joint_speed, now)
        self._mode = "joint"
        self._move_end = None

    def _move_line(self, goal, yaw, speed, now):
        start = self.kin.fk(self._q_cmd).translation.copy()
        dist = np.linalg.norm(np.asarray(goal) - start)
        self._traj = Trajectory(start, goal, 1.875 * dist / speed, now)
        self._line_rot = None if yaw is None else top_down_rotation(yaw)
        self._mode = "line"
        self._move_end = None

    def _set_gripper(self, pct, now, parallel=False):
        start = self._gripper_cmd
        traj = Trajectory([start], [pct], abs(pct - start) / self.cfg.gripper_speed, now)
        self._gripper_traj = traj
        if not parallel:
            self._traj = None
            self._mode = "gripper"
            self._move_end = None

    def _advance_motion(self, now) -> bool:
        """Update the command along the current motion. True when it is finished and settled."""
        if self._gripper_traj is not None:
            self._gripper_cmd = float(self._gripper_traj.sample(now)[0])
        if self._mode == "joint":
            self._q_cmd = self._traj.sample(now)
        elif self._mode == "line":
            point = self._traj.sample(now)
            rot = self._line_rot
            res = self.kin.solve(point, self._q_cmd, rot, soft_rotation=True, max_iters=30)
            self._q_cmd = res.q
        traj_done = (self._traj is None or self._traj.finished(now)) and \
                    (self._gripper_traj is None or self._gripper_traj.finished(now))
        if not traj_done:
            return False
        if self._move_end is None:
            self._move_end = now
        settled = now - self._move_end >= self.cfg.settle_time
        reached = self._mode == "gripper" or self.kin.joint_distance(self._q_meas, self._q_cmd) < self.cfg.reach_tolerance
        if self._mode == "gripper":
            # The gripper may be blocked by the cube: only wait for it to stop moving.
            reached = True
        if settled and reached:
            self._gripper_traj = None
            return True
        if now - self._move_end > self.cfg.move_timeout:
            self.log("warn", f"{self.state.value}: arm did not reach the command in time, continuing")
            self._gripper_traj = None
            return True
        return False

    def _command(self) -> Command:
        return Command(self.kin.joints_from_q(self._q_cmd, ARM_JOINT_NAMES), float(self._gripper_cmd))
