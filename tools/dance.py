#!/usr/bin/env python3
"""dance.py - makes the SO-ARM101 dance: a choreography of moves on a tempo, sim first, then real.

    python3 tools/dance.py --list                         # moves and choreographies
    uv run --no-project --with mujoco --with numpy --with opencv-python-headless \
        tools/dance.py --sim --video ~/Desktop/dance.mp4  # simulate + check, record a video
    python3 tools/dance.py --real                         # real arm (tools/real_arm_server.py --allow-motion)
    python3 tools/dance.py --real --bpm 80 --amp 0.6 --routine short
    ... tools/dance.py --sim --music song.mp3                # dance to a song (needs librosa, python <= 3.12)
    ... tools/dance.py --sim --music song.mp3 --video out.mp4  # video with the song on it

From Python (no CLI):  from dance import dance_to_music, choreography_from_music
    dance_to_music("song.mp3")                          # live sim + the song playing
    dance_to_music("song.mp3", mode="real", amp=0.6)    # real arm
    ch, song = choreography_from_music("song.mp3")      # just the plan: ch.q(t) -> joint targets
    (live twin, default with mujoco: http://localhost:8011, MuJoCo physics next to the real arm;
     --sim without --video plays it in real time there; --no-view to disable)

How it works: every move is a smooth periodic offset (radians) added to a base pose (arm up,
forearm forward, far from the table and from itself). Each move fades in and out over its first
and last beat, so the joint targets are continuous from one move to the next. The whole trajectory
is computed before anything moves and refused if it leaves the joint limits (with a margin), goes
faster than --vmax, or (when mujoco is installed) makes the arm touch itself or the table in a
kinematic sweep of the MuJoCo model. The gripper does not move (--gripper to clap it).

Real arm: reads the present pose, enables torque (the server holds the present pose first), goes
to the base pose slowly, dances, comes back slowly to where it started and holds it (torque stays
on: hold the arm before stopping the server). Aborts and holds the present pose on Ctrl-C or if
a joint lags its target by more than --max-lag (blocked, overloaded).
"""
import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "sim/so101_sim/assets/so101/scene_tek5.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
LIMITS = {  # rad, URDF / MJCF (same as so101_driver)
    "shoulder_pan": (-1.91986, 1.91986), "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69), "wrist_flex": (-1.65806, 1.65806), "wrist_roll": (-2.74385, 2.84121),
}
BASE = {"shoulder_pan": 0.0, "shoulder_lift": -0.25, "elbow_flex": 0.0, "wrist_flex": 0.15, "wrist_roll": 0.0}
ROBOT_BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")


def wave(beat, period, phase=0.0):
    """sin with a period in beats."""
    return math.sin(2 * math.pi * (beat / period + phase))


ON_BEAT = 0.25


def pulse(beat, period):
    """0 -> 1 -> 0 once per period (in beats): a smooth push out and back."""
    return (1 - math.cos(2 * math.pi * beat / period)) / 2


# Phase that puts the turning points (the visible "hits") of a 2-beat wave ON the beats, not between
# them: sin(2*pi*b/2) turns at b = 0.5, 1.5; with a quarter-period shift it turns at b = 0, 1, 2...
# 4-beat waves already turn on beats 1 and 3. Each move: f(beat inside the move) -> {joint: offset rad}. Amplitudes are for --amp 1.
MOVES = {
    "sway":    lambda b: {"shoulder_pan": 0.45 * wave(b, 4), "wrist_roll": -0.20 * wave(b, 4)},
    "nod":     lambda b: {"wrist_flex": 0.30 * wave(b, 2, ON_BEAT)},
    "bounce":  lambda b: {"shoulder_lift": 0.15 * wave(b, 2, ON_BEAT), "elbow_flex": -0.25 * wave(b, 2, ON_BEAT),
                          "wrist_flex": 0.12 * wave(b, 2, ON_BEAT)},
    "twist":   lambda b: {"wrist_roll": 0.48 * wave(b, 4), "shoulder_pan": 0.15 * wave(b, 4)},
    "wave":    lambda b: {"elbow_flex": 0.22 * wave(b, 2, ON_BEAT), "wrist_flex": -0.26 * wave(b, 2, ON_BEAT + 0.15)},
    "groove":  lambda b: {"shoulder_pan": 0.40 * wave(b, 4), "wrist_flex": 0.24 * wave(b, 2, ON_BEAT),
                          "wrist_roll": 0.40 * wave(b, 4, 0.25)},
    "reach":   lambda b: {"shoulder_lift": 0.35 * pulse(b, 4), "elbow_flex": -0.60 * pulse(b, 4),
                          "wrist_flex": -0.35 * pulse(b, 4)},
    "bow":     lambda b: {"shoulder_lift": 0.30 * pulse(b, 4), "wrist_flex": 0.35 * pulse(b, 4)},
    "circle":  lambda b: {"shoulder_pan": 0.40 * wave(b, 4), "shoulder_lift": 0.18 * wave(b, 4, 0.25),
                          "elbow_flex": -0.28 * wave(b, 4, 0.25)},
}
ROUTINES = {  # (move, beats); beats should be a multiple of the move's period
    "full": [("sway", 8), ("nod", 8), ("bounce", 8), ("twist", 8), ("wave", 8),
             ("circle", 8), ("groove", 16), ("reach", 8), ("bounce", 8), ("bow", 8)],
    "short": [("sway", 8), ("nod", 4), ("twist", 8), ("groove", 8), ("bow", 8)],
}
for _m in MOVES:
    ROUTINES[_m] = [(_m, 8)]


def smooth01(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (10 - 15 * x + 6 * x * x)  # min-jerk


class Choreography:
    """routine: [(move, beats)] or [(move, beats, amp)]. The beat clock is a fixed tempo (bpm) or,
    with beat_times (s, from a song), the real beats of the music: beat k happens at beat_times[k]
    (linear in between), so the moves land on the music's beats even when its tempo drifts."""

    def __init__(self, routine, bpm, amp, gripper=False, beat_times=None, duration=None):
        self.steps = [(s[0], s[1], s[2] if len(s) > 2 else 1.0) for s in routine]
        self.spb, self.amp, self.gripper = 60.0 / bpm, amp, gripper
        self.starts, t = [], 0.0
        for _, beats, _ in self.steps:
            self.starts.append(t)
            t += beats
        self.beats = t
        self.bt = None
        if beat_times is not None:
            import numpy as np
            bt = np.asarray(beat_times, dtype=float)
            self.spb = float(np.median(np.diff(bt)))
            # extrapolate the beat grid one beat before / after so beat(t) is defined everywhere
            self.bt = np.concatenate([[bt[0] - self.spb], bt, [bt[-1] + self.spb]])
            self.duration = duration if duration is not None else float(bt[-1] + self.spb)
        else:
            self.duration = t * self.spb

    def beat(self, t):
        """Beat number (float, 0 = first beat) at time t."""
        if self.bt is None:
            return t / self.spb
        import numpy as np
        return float(np.interp(t, self.bt, np.arange(-1, len(self.bt) - 1),
                               left=-1 - (self.bt[0] - t) / self.spb, right=len(self.bt) - 2 + (t - self.bt[-1]) / self.spb))

    def step_at(self, t):
        """(move name, beat inside the move, move length in beats) at time t, or (None, 0, 0)."""
        b = self.beat(t)
        for (name, beats, _), s in zip(self.steps, self.starts):
            if s <= b < s + beats:
                return name, b - s, beats
        return None, 0.0, 0

    def step_amp(self, t):
        b = self.beat(t)
        for (_, beats, a), s in zip(self.steps, self.starts):
            if s <= b < s + beats:
                return a
        return 0.0

    def move_at(self, t):
        name, bm, _ = self.step_at(t)
        return name, bm

    def q(self, t):
        """Joint targets at time t (s), gripper % included if enabled."""
        q = dict(BASE)
        name, bm, beats = self.step_at(t)
        if name:
            env = smooth01(bm) * smooth01(beats - bm)  # fade in / out over one beat
            a = self.amp * self.step_amp(t)
            for j, v in MOVES[name](bm).items():
                q[j] += a * env * v
        if self.gripper:
            q["gripper"] = 30 + 25 * (1 + math.cos(2 * math.pi * self.beat(t))) / 2  # clap on every beat
        return q


def check_limits_and_speed(ch, vmax, margin, dt=0.01):
    worst_v, n = 0.0, int(ch.duration / dt) + 1
    prev = ch.q(0.0)
    for i in range(1, n):
        q = ch.q(i * dt)
        for j in JOINTS:
            lo, hi = LIMITS[j]
            if not lo + margin <= q[j] <= hi - margin:
                return f"{j} = {q[j]:.2f} rad at t={i * dt:.1f}s leaves the limits [{lo + margin:.2f}, {hi - margin:.2f}]"
            worst_v = max(worst_v, abs(q[j] - prev[j]) / dt)
        prev = q
    if worst_v > vmax:
        return f"peak joint speed {worst_v:.2f} rad/s > --vmax {vmax}: lower --bpm or --amp"
    return None, worst_v


def check_collisions(ch, min_z=0.06, dt=0.05):
    """Kinematic sweep in the MuJoCo model: self-contacts, robot vs table/objects, TCP height."""
    try:
        import mujoco
    except ImportError:
        return "skipped (mujoco not installed)", None
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    adr = {j: m.joint(j).qposadr[0] for j in JOINTS}
    robot = {m.body(n).id for n in ROBOT_BODIES}
    lowest = 9.0
    for i in range(int(ch.duration / dt) + 1):
        q = ch.q(i * dt)
        for j in JOINTS:
            d.qpos[adr[j]] = q[j]
        mujoco.mj_forward(m, d)
        lowest = min(lowest, float(d.site("gripperframe").xpos[2]))
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 not in robot and b2 not in robot:
                continue
            if b1 in robot and b2 in robot and (m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1):
                continue
            n1, n2 = m.body(b1).name, m.body(b2).name
            if {n1, n2} & {"base"} and not ({n1, n2} - {"base", "world"}):
                continue  # the base standing on the table
            if c.dist < -0.001:
                return f"contact {n1} / {n2} at t={i * dt:.1f}s ({ch.move_at(i * dt)[0]})", lowest
    if lowest < min_z:
        return f"gripper goes down to z={lowest:.3f} m (< {min_z})", lowest
    return None, lowest


def ramp(get_q0, q1, speed, send, read=None, max_lag=None):
    """Min-jerk move in joint space from the present pose to q1 at <= speed rad/s (peak)."""
    q0 = get_q0()
    dist = max(abs(q1[j] - q0[j]) for j in JOINTS)
    T = max(1.0, 1.875 * dist / speed)  # min-jerk peak speed = 1.875 * mean
    t0 = time.monotonic()
    while (t := time.monotonic() - t0) < T:
        s = smooth01(t / T)
        send({j: q0[j] + s * (q1[j] - q0[j]) for j in JOINTS})
        time.sleep(0.02)
    send(q1)


# ---------------------------------------------------------------- simulation
def run_sim(ch, video, fps=30):
    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    act = {j: m.actuator(j).id for j in JOINTS}
    adr = {j: m.joint(j).qposadr[0] for j in JOINTS}
    for j in JOINTS:  # start in the base pose
        d.qpos[adr[j]] = d.ctrl[act[j]] = BASE[j]
    mujoco.mj_forward(m, d)
    writer = renderer = None
    if video:
        import cv2
        W, H = 960, 640
        renderer = mujoco.Renderer(m, H, W)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.05, 0.0, 0.18]
        cam.distance, cam.elevation, cam.azimuth = 0.95, -12, 150
        writer = cv2.VideoWriter(str(Path(video).expanduser()), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    lag, n_frame, t = {j: 0.0 for j in JOINTS}, 0, 0.0
    while t < ch.duration:
        q = ch.q(t)
        for j in JOINTS:
            d.ctrl[act[j]] = q[j]
        mujoco.mj_step(m, d)
        t = d.time
        for j in JOINTS:
            lag[j] = max(lag[j], abs(d.qpos[adr[j]] - q[j]))
        if writer is not None and t >= n_frame / fps:
            renderer.update_scene(d, cam)
            img = np.ascontiguousarray(renderer.render()[:, :, ::-1])
            name, bm = ch.move_at(t)
            fb = ch.beat(t)
            beat = int(math.floor(fb))
            cv2.rectangle(img, (0, 0), (W, 40), (20, 20, 20), -1)
            cv2.putText(img, f"{name or '-'}   beat {beat + 1}/{int(ch.beats)}   {t:5.1f}s", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
            if fb >= 0 and fb % 1 < 0.2:  # flash on every beat
                cv2.circle(img, (W - 30, 20), 10, (0, 200, 255), -1)
            writer.write(img)
            n_frame += 1
    if writer is not None:
        writer.release()
    print("sim: max tracking lag per joint (rad): " + ", ".join(f"{j} {v:.2f}" for j, v in lag.items()))
    if video:
        print(f"video: {Path(video).expanduser()}")


# ---------------------------------------------------------------- real arm
class Arm:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=2.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.f = self.sock.makefile("rb")

    def call(self, msg):
        self.sock.sendall((json.dumps(msg) + "\n").encode())
        r = json.loads(self.f.readline())
        if not r.get("ok", False):
            raise RuntimeError(f"real_arm_server: {r}")
        return r

    def read(self):
        o = self.call({"cmd": "read"})["obs"]
        return {j: o[f"{j}.pos"] for j in JOINTS} | {"gripper": o["gripper.pos"]}

    def send(self, q):
        self.call({"cmd": "act", "action": {f"{j}.pos": v for j, v in q.items()}})

    def hold(self):
        p = self.read()
        self.send({j: p[j] for j in JOINTS})


def start_music(args):
    """Start the song (if --music) and return (process, t0): dance time 0 = song time 0."""
    if not getattr(args, "music", None):
        return None, time.monotonic()
    import atexit
    import subprocess
    proc = subprocess.Popen(["afplay", str(Path(args.music).expanduser())])
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    return proc, time.monotonic() + args.audio_latency


def mux_music(video, music):
    """Put the song on the rendered video (same t=0)."""
    import shutil
    import subprocess
    video = Path(video).expanduser()
    tmp = video.with_suffix(".noaudio.mp4")
    shutil.move(video, tmp)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp),
                    "-i", str(Path(music).expanduser()), "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)], check=True)
    tmp.unlink()


def dance_label(ch, t):
    """'move  beat b/B  (beat n/N)' for the twin overlay."""
    name, bm, beats = ch.step_at(t)
    fb = ch.beat(t)
    if ch.bt is not None:  # dancing to a song: song clock + a pulse on every detected beat
        ts = max(t, 0.0)
        song = f"{int(ts // 60)}:{ts % 60:04.1f} / {int(ch.duration // 60)}:{ch.duration % 60:02.0f}"
        pulse = "[#####]" if fb >= 0 and fb % 1 < 0.18 else "[     ]"
        what = name or ("intro" if fb < 0 else "end")
        step = f" {int(bm) + 1}/{beats}" if name else ""
        return f"{song}  {pulse} beat {max(int(math.floor(fb)) + 1, 0)}  {what}{step}"
    if not name:
        return "intro" if fb < 0 else "end"
    return f"{name}  beat {int(bm) + 1}/{beats}   (beat {int(fb) + 1}/{int(ch.beats)})"


def open_twin(args, start=None):
    """The live twin view (tools/dance_twin.py), or None if --no-view / mujoco missing / port busy."""
    if not args.view:
        return None
    try:
        from dance_twin import Twin
    except ImportError as e:
        print(f"view disabled: {e} (run with uv ... --with mujoco --with numpy --with opencv-python-headless)")
        return None
    try:
        twin = Twin(port=args.view_port, start=start)
    except OSError as e:
        print(f"view disabled: port {args.view_port}: {e}")
        return None
    print(f"live view (left: MuJoCo sim, right: {'real arm' if args.real else 'target'}): {twin.url}")
    return twin


def run_sim_live(ch, args):
    """Real-time sim: the choreography against the wall clock, watched live in the twin."""
    twin = open_twin(args, start=BASE)
    if twin is None:
        raise SystemExit("--sim live mode needs the view: use --video, or --no-view for a headless check")
    try:
        if args.view_wait > 0:
            print(f"open the page; starting in {args.view_wait:.0f} s...")
        t_end = time.monotonic() + args.view_wait
        while time.monotonic() < t_end:  # hold the base pose while the page opens
            twin.update(BASE, None, "base pose: waiting to start")
            time.sleep(0.02)
        for k in range(args.count_in, 0, -1):
            print(f"  {k}...")
            t_end = time.monotonic() + ch.spb
            while time.monotonic() < t_end:
                twin.update(BASE, None, f"count-in {k}...")
                time.sleep(0.02)
        print(f"dancing (sim, real time): {ch.duration:.0f} s at {60 / ch.spb:.0f} bpm")
        audio, t0 = start_music(args)
        shown = None
        while (t := time.monotonic() - t0) < ch.duration:
            q = ch.q(t)
            twin.update(q, None, dance_label(ch, t))
            name = ch.move_at(t)[0]
            if name != shown:
                print(f"  [{t:5.1f}s] {name}")
                shown = name
            time.sleep(max(0.0, 0.02 - (time.monotonic() - t0 - t)))
        t_end = time.monotonic() + 2.0
        while time.monotonic() < t_end:
            twin.update(BASE, None, "done")
            time.sleep(0.02)
        dev = twin.max_deviation()
        print("sim: max |sim - target| per joint (rad): " + ", ".join(f"{j} {v:.2f}" for j, v in dev.items()))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        twin.close()


def run_real(ch, args):
    arm = Arm(args.host, args.port)
    if not arm.call({"cmd": "info"}).get("allow_motion", True):
        raise SystemExit("real_arm_server runs READ-ONLY: restart it with --allow-motion")
    start = arm.read()
    print("start pose: " + ", ".join(f"{j} {start[j]:+.2f}" for j in JOINTS))
    twin = open_twin(args, start=start)
    arm.call({"cmd": "torque", "on": True})
    lagged = {"n": 0}

    def send_measured(q, label):
        """Send q, read the arm back once, feed the twin. Returns the measured pose."""
        arm.send(q)
        p = arm.read()
        twin.update(q, p, label)
        return p

    def send_ramp(label):  # ramps: plain send without the view (no extra read), send + read with it
        return arm.send if twin is None else (lambda q: send_measured(q, label))

    def send_checked(q, label=""):
        if twin is None:
            arm.send(q)
            p = arm.read()
        else:
            p = send_measured(q, label)
        worst = max(abs(p[j] - q[j]) for j in JOINTS)
        lagged["n"] = lagged["n"] + 1 if worst > args.max_lag else 0
        if lagged["n"] >= 8:  # ~0.3 s behind the target
            raise RuntimeError(f"arm lags its target by {worst:.2f} rad (blocked?)")

    try:
        print("going to the base pose...")
        ramp(arm.read, BASE, args.approach_speed, send_ramp("approach: to the base pose"))

        def hold_base(seconds, label, settle_tol=None):
            """Keep commanding BASE (the server freezes the arm after 1 s of silence). With settle_tol,
            return as soon as every joint is within it, and abort if it never gets there."""
            t_end = time.monotonic() + seconds
            while time.monotonic() < t_end:
                arm.send(BASE)
                p = arm.read()
                if twin is not None:
                    twin.update(BASE, p, label)
                worst = max(abs(p[j] - BASE[j]) for j in JOINTS)
                if settle_tol is not None and worst <= settle_tol:
                    return
                time.sleep(0.02)
            if settle_tol is not None:
                raise RuntimeError(f"base pose not reached: still {worst:.2f} rad away after the approach "
                                   "(joint too weak/slow? try real_arm_server --max-step 0.15)")

        hold_base(args.settle_time, "settling at the base pose", settle_tol=args.settle_tol)
        for k in range(args.count_in, 0, -1):
            print(f"  {k}...")
            hold_base(ch.spb, f"count-in {k}")
        print(f"dancing: {ch.duration:.0f} s at {60 / ch.spb:.0f} bpm")
        audio, t0 = start_music(args)
        shown = None
        while (t := time.monotonic() - t0) < ch.duration:
            q = ch.q(t)
            send_checked({j: q[j] for j in JOINTS} | ({"gripper": q["gripper"]} if "gripper" in q else {}),
                         dance_label(ch, t) if twin else "")
            name = ch.move_at(t)[0]
            if name != shown:
                print(f"  [{t:5.1f}s] {name}")
                shown = name
            time.sleep(max(0.0, 0.02 - (time.monotonic() - t0 - t)))
        print("back to the start pose...")
        ramp(arm.read, {j: start[j] for j in JOINTS}, args.approach_speed, send_ramp("return: to the start pose"))
        print("done: the arm holds its start pose (torque on). Hold it before stopping the server.")
        if twin is not None:
            dev = twin.max_deviation()
            print("twin: max |sim - real| per joint (rad): " + ", ".join(f"{j} {v:.2f}" for j, v in dev.items()))
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\nABORT ({str(e) or 'Ctrl-C'}): holding the present pose")
        arm.hold()
        sys.exit(1)
    finally:
        if twin is not None:
            twin.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sim", action="store_true", help="run in MuJoCo (default)")
    g.add_argument("--real", action="store_true", help="run on the real arm through real_arm_server")
    g.add_argument("--list", action="store_true", help="list moves and routines")
    p.add_argument("--routine", default="full", help="full, short, or one move name")
    p.add_argument("--bpm", type=float, default=90.0)
    p.add_argument("--amp", type=float, default=1.0, help="amplitude scale (0.5 = half as wide)")
    p.add_argument("--gripper", action="store_true", help="clap the gripper on the beat (needs a good calibration)")
    p.add_argument("--vmax", type=float, default=2.0, help="refuse above this joint speed (rad/s)")
    p.add_argument("--margin", type=float, default=0.15, help="distance kept from the joint limits (rad)")
    p.add_argument("--video", help="--sim: write an mp4 here")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--approach-speed", type=float, default=0.4, help="rad/s to reach / leave the base pose")
    p.add_argument("--max-lag", type=float, default=0.35, help="abort if a joint lags its target this much (rad)")
    p.add_argument("--count-in", type=int, default=4, help="beats counted before dancing")
    p.add_argument("--settle-time", type=float, default=5.0, help="max s to reach the base pose after the approach")
    p.add_argument("--settle-tol", type=float, default=0.08, help="rad: base pose considered reached")
    p.add_argument("--view", action=argparse.BooleanOptionalAction, default=True,
                   help="live twin in the browser: MuJoCo sim next to the real arm (--sim: next to the "
                        "target, in real time unless --video); --no-view to disable (no mujoco needed)")
    p.add_argument("--view-port", type=int, default=8011, help="port of the live view")
    p.add_argument("--view-wait", type=float, default=3.0, help="--sim live: seconds before starting")
    p.add_argument("--music", help="dance to this song (mp3/wav): beats + loudness -> choreography, played live")
    p.add_argument("--audio-latency", type=float, default=0.1, help="--music: s between starting the player "
                   "and the song's t=0 (raise it if the arm is ahead of the sound)")
    p.add_argument("--max-dance-bpm", type=float, default=115.0, help="--music: faster songs are danced half time")
    args = p.parse_args(argv)

    if args.list:
        print("moves:    " + ", ".join(MOVES))
        for name, steps in ROUTINES.items():
            if name not in MOVES:
                print(f"routine {name}: " + " -> ".join(f"{m}({b})" for m, b in steps))
        return
    if args.routine not in ROUTINES:
        raise SystemExit(f"unknown routine {args.routine!r}: --list")
    if args.music:
        import dance_music
        song = dance_music.analyze(Path(args.music).expanduser(), args.max_dance_bpm)
        routine = dance_music.build_routine(song["energy"], Path(args.music).name)
        print(f"song: {song['duration']:.0f} s, {song['tempo']:.0f} bpm"
              + (" (danced half time)" if song["half_time"] else "") + f", {len(song['beat_times'])} beats, "
              f"first beat at {song['beat_times'][0]:.2f} s")
        print("choreography: " + " ".join(f"{m}({a:.2f})" for m, _, a in routine))
        args.count_in = 0
        make = lambda amp: Choreography(routine, args.bpm, amp, args.gripper,  # noqa: E731
                                        beat_times=song["beat_times"], duration=song["duration"])
    else:
        make = lambda amp: Choreography(ROUTINES[args.routine], args.bpm, amp, args.gripper)  # noqa: E731
    ch = make(args.amp)
    res = check_limits_and_speed(ch, args.vmax, args.margin)
    if args.music and isinstance(res, str) and "speed" in res:  # a fast song: shrink the moves to fit
        peak = float(res.split("speed ")[1].split(" ")[0])
        ch = make(args.amp * 0.95 * args.vmax / peak)
        print(f"amplitude scaled to {ch.amp:.2f} to stay under --vmax {args.vmax}")
        res = check_limits_and_speed(ch, args.vmax, args.margin)
    if isinstance(res, str):
        raise SystemExit(f"REFUSED: {res}")
    print(f"{'song' if args.music else 'routine ' + args.routine}: {ch.duration:.0f} s, {int(ch.beats)} beats "
          f"at {60 / ch.spb:.0f} bpm, peak joint speed {res[1]:.2f} rad/s")
    err, lowest = check_collisions(ch)
    if err and not err.startswith("skipped"):
        raise SystemExit(f"REFUSED: {err}")
    print(f"collision check: {err or 'OK'}" + (f", gripper never below z={lowest:.3f} m" if lowest else ""))

    if args.real:
        run_real(ch, args)
    elif args.view and not args.video:
        run_sim_live(ch, args)
    else:
        run_sim(ch, args.video)
        if args.video and args.music:
            mux_music(args.video, args.music)
            print(f"video with the song: {Path(args.video).expanduser()}")


# ---------------------------------------------------------------- Python API (no CLI needed)
def choreography_from_music(music, amp=1.0, max_dance_bpm=115.0, gripper=False):
    """Analyse a song and return (Choreography, song info). Pure computation, nothing moves:
    ch.q(t) gives the joint targets (rad) at song time t, ch.beat(t) the beat number."""
    import dance_music
    song = dance_music.analyze(Path(music).expanduser(), max_dance_bpm)
    routine = dance_music.build_routine(song["energy"], Path(music).name)
    song["routine"] = routine
    return Choreography(routine, 90.0, amp, gripper, beat_times=song["beat_times"],
                        duration=song["duration"]), song


def dance_to_music(music, mode="sim", video=None, amp=1.0, view=True, host="127.0.0.1", port=5555,
                   **options):
    """Make the arm dance to a song.

    mode="sim":  live MuJoCo view on http://localhost:8011 with the song playing (or, with
                 video="out.mp4", render a video with the song instead).
    mode="real": the real arm through tools/real_arm_server.py --allow-motion (sim twin alongside).
    options: any other dance.py flag, e.g. vmax=1.5, audio_latency=0.15, view_port=8012.
    Same checks as the CLI (limits, speed, collisions) before anything moves."""
    argv = ["--" + mode, "--music", str(music), "--amp", str(amp), "--host", host, "--port", str(port)]
    if video:
        argv += ["--video", str(video)]
    if not view:
        argv.append("--no-view")
    for k, v in options.items():
        flag = "--" + k.replace("_", "-")
        argv += [flag] if v is True else ([] if v is False else [flag, str(v)])
    main(argv)


if __name__ == "__main__":
    main()
