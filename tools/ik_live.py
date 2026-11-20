#!/usr/bin/env python3
"""ik_live.py - live view of the IK test in the browser: the arm reaches random XYZ targets.

    python3 tools/ik_live.py                 # -> http://localhost:8009
    python3 tools/ik_live.py --hard 0.5      # half of the targets near the base / low (self-collision zone)
    python3 tools/ik_live.py --no-self-collision   # start with the check off (toggle in the page)

Same measure as `eval_pick_place.py --ik-test`: IK (so101_brain.kinematics) -> joint command ->
MuJoCo, then the TCP really reached is measured in the simulation. Real time, smooth joint-space
moves. The page shows the 3D view (target sphere, TCP trail), the IK verdict, the clearance
between non-adjacent links, the self-contacts MuJoCo sees and the running statistics, and
toggles the self-collision check of the IK live to compare.
Needs mujoco, pinocchio (pip: pin), opencv, numpy.
"""
import argparse
import http.server
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "ros2_ws/src/so101_brain")]

from so101_sim import SO101Sim  # noqa: E402

from so101_brain.kinematics import ArmKinematics  # noqa: E402

SCENE = ROOT / "sim/so101_sim/assets/so101/scene_tek5.xml"
URDF = ROOT / "ros2_ws/src/so101_description/urdf/so101.urdf"
W, H = 960, 640
ROBOT_BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>IK live</title><style>
body{background:#111;color:#ddd;font:14px system-ui;margin:16px} h1{font-size:18px;margin:0 0 10px}
.wrap{display:flex;gap:16px;flex-wrap:wrap} img{max-width:100%;border:1px solid #333;border-radius:4px}
.panel{min-width:280px;max-width:420px;flex:1} .v{font-size:20px;font-weight:600;margin:6px 0}
.ok{color:#6c6} .warn{color:#e9a23b} .bad{color:#ef5350} table{border-collapse:collapse;width:100%}
td{padding:3px 6px;border-bottom:1px solid #2a2a2a} td:last-child{text-align:right;font-variant-numeric:tabular-nums}
button{background:#263238;color:#ddd;border:1px solid #455a64;border-radius:4px;padding:6px 10px;margin:4px 4px 4px 0;cursor:pointer}
button.on{background:#1b5e20;border-color:#2e7d32} button.off{background:#5d1b1b;border-color:#8e2b2b}
#log{font:12px ui-monospace,monospace;color:#aaa;max-height:240px;overflow:auto;white-space:pre}
</style></head><body><h1>SO-ARM101 &mdash; IK live (MuJoCo, error measured in the simulation)</h1>
<div class="wrap"><img src="stream.mjpg" width="960" height="640" alt="3D view">
<div class="panel">
<button id="sc"></button><button id="hard"></button><button id="pause"></button>
<div class="v" id="verdict">&hellip;</div>
<table id="now"></table><h1 style="margin-top:14px">Statistics</h1><table id="stats"></table>
<h1 style="margin-top:14px">Last targets</h1><div id="log"></div></div></div>
<script>
const rows = (id, kv) => document.getElementById(id).innerHTML =
  kv.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
async function post(what){ await fetch(what, {method: "POST"}); }
document.getElementById("sc").onclick = () => post("toggle_self_collision");
document.getElementById("hard").onclick = () => post("toggle_hard");
document.getElementById("pause").onclick = () => post("toggle_pause");
async function tick(){
  try {
    const s = await (await fetch("status.json?" + Date.now())).json();
    const b = document.getElementById("sc");
    b.textContent = "Self-collision check: " + (s.self_collision ? "ON" : "OFF");
    b.className = s.self_collision ? "on" : "off";
    document.getElementById("hard").textContent = "Hard targets: " + Math.round(100 * s.hard) + " %";
    document.getElementById("pause").textContent = s.paused ? "Resume" : "Pause";
    const v = document.getElementById("verdict"); v.textContent = s.verdict; v.className = "v " + s.level;
    rows("now", s.now); rows("stats", s.stats);
    document.getElementById("log").textContent = s.log.join("\\n");
  } catch (e) {}
  setTimeout(tick, 300);
}
tick();
</script></body></html>"""


class Live:
    """Shared state between the simulation loop (main thread, owns OpenGL) and the HTTP server."""

    def __init__(self, hard, self_collision):
        self.lock = threading.Lock()
        self.jpeg = None
        self.frame_id = 0
        self.status = {}
        self.hard = hard
        self.self_collision = self_collision
        self.paused = False


def make_handler(live: Live):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/status.json":
                with live.lock:
                    body = json.dumps(live.status).encode()
                self._send(body, "application/json")
            elif path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = -1
                try:
                    while True:
                        with live.lock:
                            jpeg, fid = live.jpeg, live.frame_id
                        if jpeg is None or fid == last:
                            time.sleep(0.01)
                            continue
                        last = fid
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            with live.lock:
                if self.path == "/toggle_self_collision":
                    live.self_collision = not live.self_collision
                elif self.path == "/toggle_hard":
                    live.hard = {0.0: 0.5, 0.5: 1.0}.get(live.hard, 0.0)
                elif self.path == "/toggle_pause":
                    live.paused = not live.paused
                else:
                    self.send_error(404)
                    return
            self._send(b"ok", "text/plain")

    return Handler


def random_target(rng, hard):
    """Same distribution as --ik-test; a 'hard' target is near the base or very low."""
    if rng.random() < hard:
        if rng.random() < 0.5:
            r, z = rng.uniform(0.12, 0.16), rng.uniform(0.02, 0.08)
        else:
            r, z = rng.uniform(0.18, 0.24), rng.uniform(0.02, 0.045)
    else:
        r, z = rng.uniform(0.12, 0.32), rng.uniform(0.02, 0.20)
    a = rng.uniform(-1.2, 1.2)
    return np.array([r * np.cos(a), r * np.sin(a), z])


def self_contacts(model, data):
    """Robot body pairs in contact in MuJoCo (the arm touching itself), adjacent bodies excluded."""
    pairs = set()
    for c in data.contact[:data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        n1, n2 = model.body(b1).name, model.body(b2).name
        if n1 in ROBOT_BODIES and n2 in ROBOT_BODIES and model.body_parentid[b1] != b2 \
                and model.body_parentid[b2] != b1:
            pairs.add(" / ".join(sorted((n1, n2))))
    return sorted(pairs)


def add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0]), np.asarray(pos, dtype=float), np.eye(3).flatten(),
                        np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def min_jerk(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--hard", type=float, default=0.0, help="share of targets in the self-collision zone")
    ap.add_argument("--no-self-collision", action="store_true", help="start with the IK check off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()

    live = Live(args.hard, not args.no_self_collision)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(live))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"IK live: http://localhost:{args.port}  (Ctrl+C to stop)")

    rng = np.random.default_rng(args.seed)
    sim = SO101Sim(scene_path=SCENE, seed=args.seed)
    sim.connect()
    m, d = sim.model, sim.data
    kins = {True: ArmKinematics(URDF, seed=args.seed),
            False: ArmKinematics(URDF, seed=args.seed, self_collision=False)}
    if not kins[True].checks_self_collision:
        print("warning: collision meshes not loaded, the check cannot be turned on")
    renderer = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.10, 0.0, 0.08]
    cam.distance, cam.elevation, cam.azimuth = 0.85, -25, 200

    stats = {True: [], False: []}          # measured errors (mm) per check mode
    refused = {True: [0, 0], False: [0, 0]}  # [unreachable, self-collision]
    stalled = {True: 0, False: 0}          # reached by the IK but MuJoCo saw self-contacts
    history = deque(maxlen=12)
    trail = deque(maxlen=90)
    q = kins[True].neutral()
    period = 1.0 / args.fps
    n = 0

    def publish(verdict, level, now_rows):
        def stat_rows(mode):
            e = np.array(stats[mode]) if stats[mode] else None
            name = "ON" if mode else "OFF"
            out = [(f"[check {name}] targets reached", f"{len(stats[mode])}"),
                   (f"[check {name}] refused: unreachable / self-collision", f"{refused[mode][0]} / {refused[mode][1]}")]
            if e is not None:
                out += [(f"[check {name}] error mean / median", f"{e.mean():.2f} / {np.median(e):.2f} mm"),
                        (f"[check {name}] error max", f"{e.max():.2f} mm"),
                        (f"[check {name}] under 1 cm", f"{np.mean(e < 10) * 100:.0f} %"),
                        (f"[check {name}] arm hit itself in MuJoCo", f"{stalled[mode]}")]
            return out
        with live.lock:
            live.status = {"self_collision": live.self_collision, "hard": live.hard, "paused": live.paused,
                           "verdict": verdict, "level": level, "now": now_rows,
                           "stats": stat_rows(True) + stat_rows(False), "log": list(history)}

    def render(target, colour, banner):
        renderer.update_scene(d, cam)
        add_sphere(renderer.scene, target, 0.012, colour)
        for i, p in enumerate(trail):
            add_sphere(renderer.scene, p, 0.003, (0.3, 0.7, 1.0, 0.25 + 0.6 * i / max(1, len(trail))))
        img = np.ascontiguousarray(renderer.render()[:, :, ::-1])
        cv2.rectangle(img, (0, 0), (W, 34), (20, 20, 20), -1)
        cv2.putText(img, banner, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
        ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with live.lock:
            live.jpeg, live.frame_id = jpeg.tobytes(), live.frame_id + 1

    def run_for(seconds, step_fn, target, colour, banner):
        """Advance the simulation in real time for `seconds` (paused time does not count)."""
        t, last = 0.0, time.perf_counter()
        while t < seconds:
            with live.lock:
                paused = live.paused
            if not paused:
                step_fn(t)
                t += 0.01
                trail.append(d.site("gripperframe").xpos.copy())
            if paused or time.perf_counter() - last >= period:
                render(target, colour, banner)
                last = time.perf_counter()
                time.sleep(period if paused else 0.0)
            else:
                time.sleep(max(0.0, 0.01 - 0.002))

    try:
        while True:
            with live.lock:
                mode, hard = live.self_collision, live.hard
            kin = kins[mode]
            target = random_target(rng, hard)
            n += 1
            t0 = time.perf_counter()
            res = kin.solve_global(target, q, frame="tcp")
            ik_ms = (time.perf_counter() - t0) * 1000
            # the clearance of the solution as the check would see it, even with the check off
            clearance = kins[True].self_clearance(res.q) * 1000 if kins[True].checks_self_collision else float("nan")
            tag = f"#{n} [{'ON' if mode else 'OFF'}] {np.round(target, 3).tolist()}"
            base_rows = [("target (m)", ", ".join(f"{v:.3f}" for v in target)),
                         ("radius / height", f"{np.hypot(*target[:2]):.3f} / {target[2]:.3f} m"),
                         ("IK time", f"{ik_ms:.0f} ms"),
                         ("IK residual", f"{res.position_error * 1000:.2f} mm"),
                         ("clearance between links", f"{clearance:.1f} mm")]
            if not res.ok(1e-3):
                kind = "self-collision" if res.self_collision else "unreachable"
                refused[mode][res.self_collision] += 1
                verdict, level = f"Refused by the IK: {kind}", "warn"
                history.appendleft(f"{tag}  refused ({kind})")
                publish(verdict, level, base_rows)
                run_for(1.2, lambda t: sim.send_action(
                    {f"{k}.pos": v for k, v in kin.joints_from_q(q).items()}), target, (1.0, 0.6, 0.1, 0.9),
                    f"{tag}  refused: {kind}")
                continue
            start, goal = kin.joints_from_q(q), kin.joints_from_q(res.q)
            dist = max(abs(goal[k] - start[k]) for k in goal)
            duration = max(0.8, dist / 1.2)  # ~1.2 rad/s
            publish("Moving...", "ok", base_rows)
            seen = set()

            def step(t):
                s = min_jerk(t / duration)
                sim.send_action({f"{k}.pos": start[k] + (goal[k] - start[k]) * s for k in goal})
                seen.update(self_contacts(m, d))

            run_for(duration + 0.6, step, target, (0.2, 0.9, 0.3, 0.9), f"{tag}  moving")
            q = res.q
            err = float(np.linalg.norm(d.site("gripperframe").xpos - target) * 1000)
            hit = sorted(seen)
            stats[mode].append(err)
            stalled[mode] += bool(hit)
            if hit:
                verdict, level = f"Arm hit itself: error {err:.1f} mm", "bad"
            elif err < 10:
                verdict, level = f"Reached: error {err:.2f} mm", "ok"
            else:
                verdict, level = f"Missed: error {err:.1f} mm", "bad"
            history.appendleft(f"{tag}  error {err:6.2f} mm" + (f"  HIT: {', '.join(hit)}" if hit else ""))
            rows = base_rows + [("error measured in MuJoCo", f"{err:.2f} mm"),
                                ("self-contacts during the move", ", ".join(hit) if hit else "none")]
            publish(verdict, level, rows)
            colour = (0.95, 0.25, 0.25, 0.9) if (hit or err >= 10) else (0.2, 0.9, 0.3, 0.9)
            run_for(0.9, lambda t: sim.send_action({f"{k}.pos": v for k, v in goal.items()}), target, colour,
                    f"{tag}  error {err:.2f} mm" + ("  (arm hit itself)" if hit else ""))
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
