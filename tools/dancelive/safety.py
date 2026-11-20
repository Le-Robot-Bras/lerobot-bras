"""Safety: the analytic speed / acceleration budget of a sum of move components, the per-joint
velocity + acceleration limiter with joint limits, the MuJoCo contact guard, and the offline
check of every move (--check-moves)."""
import math

import numpy as np

import dance
from dance import JOINTS, LIMITS, MOVES, ROBOT_BODIES, SCENE

from .moves import ALL_MOVES, MOVE_SPEED, mirror


def budget_scale(comps, bps, vmax, amax):
    """Speed scale k <= 1 keeping the summed peak speed / acceleration per joint of the weighted
    components (weight, offsets, (peak, d/dbeat, d2/dbeat2)) under 85 % of vmax / amax at `bps` beats/s."""
    spd = np.zeros(len(JOINTS))
    acc = np.zeros(len(JOINTS))
    for wgt, _, (mx, d1, d2) in comps:
        spd += abs(wgt) * (d1 * bps + mx * 1.9 * bps)  # + crossfade / envelope slopes
        acc += abs(wgt) * (d2 * bps * bps + mx * 6.0 * bps * bps)
    return min(1.0, 0.85 * vmax / max(spd.max(), 1e-9), 0.85 * amax / max(acc.max(), 1e-9))


class SafetyFilter:
    """Per-joint velocity + acceleration limiter (brakes in time), joint limits with a margin."""

    def __init__(self, q0, vmax=1.5, amax=12.0, margin=0.15):
        self.q = {j: float(q0[j]) for j in JOINTS}
        self.v = {j: 0.0 for j in JOINTS}
        self.vmax, self.amax, self.margin = vmax, amax, margin

    def clamp(self, j, x):
        lo, hi = LIMITS[j]
        return min(max(x, lo + self.margin), hi - self.margin)

    def propose(self, target, dt):
        q, v = {}, {}
        for j in JOINTS:
            err = self.clamp(j, target[j]) - self.q[j]
            vd = math.copysign(min(abs(err) / dt, self.vmax, math.sqrt(2.0 * self.amax * abs(err))), err)
            vj = self.v[j] + min(max(vd - self.v[j], -self.amax * dt), self.amax * dt)
            vj = min(max(vj, -self.vmax), self.vmax)
            qj = self.clamp(j, self.q[j] + vj * dt)
            q[j], v[j] = qj, (qj - self.q[j]) / dt
        return q, v

    def commit(self, q, v):
        self.q, self.v = dict(q), dict(v)

    def stop(self):
        self.v = {j: 0.0 for j in JOINTS}


class ContactGuard:
    """MuJoCo kinematic check of one pose: self contacts, robot vs table, gripper height
    (same rules as dance.check_collisions)."""

    def __init__(self, min_z=0.06):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.adr = {j: self.m.joint(j).qposadr[0] for j in JOINTS}
        self.robot = {self.m.body(n).id for n in ROBOT_BODIES}
        self.min_z = min_z

    def check(self, q):
        m, d = self.m, self.d
        for j in JOINTS:
            d.qpos[self.adr[j]] = q[j]
        self.mj.mj_forward(m, d)
        z = float(d.site("gripperframe").xpos[2])
        if z < self.min_z:
            return f"gripper at z={z:.3f} m"
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 not in self.robot and b2 not in self.robot:
                continue
            if b1 in self.robot and b2 in self.robot and (m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1):
                continue
            n1, n2 = m.body(b1).name, m.body(b2).name
            if {n1, n2} & {"base"} and not ({n1, n2} - {"base", "world"}):
                continue
            if c.dist < -0.001:
                return f"contact {n1} / {n2}"
        return None


def check_moves(bpm=130.0, vmax=3.5, margin=0.15):
    """Every move (dance.MOVES + STYLE_MOVES, each also mirrored left/right) at amp 1 through
    dance.py's offline checks: joint limits (with the margin) and collisions (kinematic MuJoCo
    sweep); also reports the peak joint speed at `bpm` and the tempo up to which it stays under
    85 % of `vmax` at full amplitude (above it the live dancer scales it down analytically)."""
    ok = True
    saved = dict(MOVES)
    try:
        for name in ALL_MOVES:
            for mir in (False, True):
                f = ALL_MOVES[name]
                MOVES["_check"] = (lambda b, f=f: mirror(f(b))) if mir else f
                ch = dance.Choreography([("_check", 8), ("_check", 8)], bpm, 1.0)
                res = dance.check_limits_and_speed(ch, 1e9, margin)
                err, lowest = dance.check_collisions(ch)
                good = not isinstance(res, str) and not (err and not err.startswith("skipped"))
                ok &= good
                full = 60.0 * 0.85 * vmax / MOVE_SPEED[name]
                print(f"  {name + ('~' if mir else ''):10s} {'OK  ' if good else 'FAIL'} "
                      + (res if isinstance(res, str) else f"limits ok, peak {res[1]:.2f} rad/s at {bpm:.0f} bpm")
                      + f", collisions: {err or 'none'}" + (f", gripper z >= {lowest:.3f} m" if lowest else "")
                      + f", full amplitude up to {full:.0f} bpm")
    finally:
        MOVES.clear()
        MOVES.update(saved)
    return ok
