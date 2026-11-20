#!/usr/bin/env python3
"""dance_twin.py - live digital twin of the SO-ARM101 in the browser, used by tools/dance.py --view.

    twin = Twin(port=8011)                  # -> http://localhost:8011
    twin.update(target, real, "sway  beat 3/32")   # every control tick, radians, plain joint names
    twin.close()

One image, two arms side by side (same model, same camera):
  LEFT  "SIM"    a MuJoCo physics simulation whose position actuators receive the same joint targets
                 as the real arm, at the time they were sent (sample-and-hold, like the servos), and
                 is stepped with mj_step up to the wall clock: a second robot executing the commands.
  RIGHT "REAL"   the measured joint positions of the real arm (qpos + mj_forward, no physics), or
        "TARGET" the commanded pose when there is no real arm (real=None).
The overlay shows the label (move, beat), the time and the per-joint difference SIM - REAL in degrees;
the page also shows target / sim / real per joint and the max deviations so far.

update() only takes a lock and appends to a queue: it never waits on rendering. Physics, rendering
(~25 fps, offscreen OpenGL) and JPEG encoding run in the twin's own thread, HTTP in another one.
Endpoints: /  /stream.mjpg  /shot.jpg (latest frame)  /status.json
Needs mujoco, numpy, opencv.
"""
import http.server
import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "sim/so101_sim/assets/so101/scene_tek5.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SHORT = {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elbow",
         "wrist_flex": "wflex", "wrist_roll": "roll"}
PW, PH = 640, 540  # one pane

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Dance twin</title><style>
body{background:#111;color:#ddd;font:14px system-ui;margin:16px} h1{font-size:18px;margin:0 0 10px}
.wrap{display:flex;gap:16px;flex-wrap:wrap} img{max-width:100%;height:auto;border:1px solid #333;border-radius:4px}
.panel{min-width:300px;max-width:520px;flex:1} .v{font-size:18px;font-weight:600;margin:6px 0}
table{border-collapse:collapse;width:100%} th{text-align:right;color:#999;font-weight:500;padding:3px 6px}
th:first-child{text-align:left} td{padding:3px 6px;border-bottom:1px solid #2a2a2a;text-align:right;
font-variant-numeric:tabular-nums} td:first-child{text-align:left} .ok{color:#6c6} .warn{color:#e9a23b} .bad{color:#ef5350}
</style></head><body><h1>SO-ARM101 &mdash; dance digital twin (left: MuJoCo physics, right: <span id="rl">real arm</span>)</h1>
<div class="wrap"><img src="stream.mjpg" width="1280" height="540" alt="twin view">
<div class="panel"><div class="v" id="label">waiting for the first command&hellip;</div>
<div id="info"></div>
<table id="joints"></table>
<p style="color:#888;font-size:12px">degrees. dev = SIM &minus; <span id="rl2">REAL</span> now;
max dev = largest |SIM &minus; <span id="rl3">REAL</span>| so far; max lag = largest |REAL &minus; TARGET| so far.</p>
</div></div>
<script>
const cls = v => Math.abs(v) < 3 ? "ok" : Math.abs(v) < 8 ? "warn" : "bad";
const f = v => v === null ? "&ndash;" : v.toFixed(1);
async function tick(){
  try {
    const s = await (await fetch("status.json?" + Date.now())).json();
    document.getElementById("label").textContent = s.label || "";
    const rn = s.has_real ? "REAL" : "TARGET";
    document.getElementById("rl").textContent = s.has_real ? "real arm (measured)" : "target (no real arm)";
    for (const id of ["rl2","rl3"]) document.getElementById(id).textContent = rn;
    document.getElementById("info").innerHTML = `t = ${s.t.toFixed(1)} s &middot; render ${s.fps.toFixed(0)} fps`
      + ` &middot; max dev ${f(s.max_dev_all)}&deg;` + (s.has_real ? ` &middot; max lag ${f(s.max_lag_all)}&deg;` : "");
    let h = `<tr><th>joint</th><th>target</th><th>sim</th><th>${rn.toLowerCase()}</th><th>dev</th><th>max dev</th>`
      + (s.has_real ? "<th>max lag</th>" : "") + "</tr>";
    for (const j of s.joints) {
      h += `<tr><td>${j.name}</td><td>${f(j.target)}</td><td>${f(j.sim)}</td><td>${f(j.real)}</td>`
        + `<td class="${cls(j.dev)}">${f(j.dev)}</td><td class="${cls(j.max_dev)}">${f(j.max_dev)}</td>`
        + (s.has_real ? `<td class="${cls(j.max_lag)}">${f(j.max_lag)}</td>` : "") + "</tr>";
    }
    document.getElementById("joints").innerHTML = h;
  } catch (e) {}
  setTimeout(tick, 250);
}
tick();
</script></body></html>"""


class Twin:
    def __init__(self, port=8011, start=None, fps=25.0, scene=SCENE, host="127.0.0.1"):
        """start: optional {joint: rad} pose to put the sim arm in before the first update()."""
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d_sim = mujoco.MjData(self.m)
        self.d_real = mujoco.MjData(self.m)
        self.adr = {j: self.m.joint(j).qposadr[0] for j in JOINTS + ["gripper"]}
        self.act = {j: self.m.actuator(j).id for j in JOINTS + ["gripper"]}
        self.g_lo, self.g_hi = self.m.actuator_ctrlrange[self.act["gripper"]]
        self.fps = fps
        self.lock = threading.Lock()
        self.cmds = deque()  # (monotonic time, target) not yet applied to the physics
        self.start = dict(start) if start else None
        self.t0 = None  # monotonic time of the first update (sim time 0)
        self.target = self.real = None
        self.label = ""
        self.jpeg, self.frame_id, self.status = None, 0, {"t": 0.0, "fps": 0.0, "label": "", "joints": [],
                                                         "has_real": False, "max_dev_all": None,
                                                         "max_lag_all": None}
        self.max_dev = {j: 0.0 for j in JOINTS}
        self.max_lag = {j: 0.0 for j in JOINTS}
        self._stop = threading.Event()
        self.srv = http.server.ThreadingHTTPServer((host, port), _make_handler(self))
        self.srv.daemon_threads = True
        self.url = f"http://localhost:{port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True, name="twin-http").start()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="twin-render")
        self._thread.start()

    # ------------------------------------------------------------ caller side (never blocks long)
    def update(self, target, real=None, label=""):
        """target / real: {joint: rad} (gripper, if present, in % 0-100 like real_arm_server)."""
        now = time.monotonic()
        with self.lock:
            if self.t0 is None:
                self.t0 = now
            self.cmds.append((now, dict(target)))
            self.target = dict(target)
            self.real = dict(real) if real is not None else None
            self.label = label

    def max_deviation(self):
        """{joint: max |SIM - REAL (or TARGET)| rad} so far."""
        with self.lock:
            return dict(self.max_dev)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.srv.shutdown()
        self.srv.server_close()

    # ------------------------------------------------------------ twin thread
    def _gripper_rad(self, pct):
        return self.g_lo + (self.g_hi - self.g_lo) * min(max(pct, 0.0), 100.0) / 100.0

    def _set_pose(self, d, q):
        for j in JOINTS:
            if j in q:
                d.qpos[self.adr[j]] = q[j]
        if "gripper" in q:
            d.qpos[self.adr["gripper"]] = self._gripper_rad(q["gripper"])

    def _set_ctrl(self, q):
        for j in JOINTS:
            if j in q:
                self.d_sim.ctrl[self.act[j]] = q[j]
        if "gripper" in q:
            self.d_sim.ctrl[self.act["gripper"]] = self._gripper_rad(q["gripper"])

    def _loop(self):
        m, ds, dr = self.m, self.d_sim, self.d_real
        renderer = mujoco.Renderer(m, PH, PW)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.05, 0.0, 0.17]
        cam.distance, cam.elevation, cam.azimuth = 0.70, -18, 160
        initialised, period, last, fps_ema = False, 1.0 / self.fps, time.monotonic(), self.fps
        while not self._stop.is_set():
            now = time.monotonic()
            with self.lock:
                t0, target, real, label = self.t0, self.target, self.real, self.label
                cmds = list(self.cmds)
                self.cmds.clear()
            if t0 is None:  # nothing yet: show the start pose (or the model default)
                if self.start:
                    self._set_pose(ds, self.start)
                    self._set_pose(dr, self.start)
                mujoco.mj_forward(m, ds)
                mujoco.mj_forward(m, dr)
            else:
                if not initialised:  # sim arm starts in the start pose, else the first target
                    q_init = self.start or cmds[0][1]
                    self._set_pose(ds, q_init)
                    self._set_ctrl(q_init)
                    ds.time = 0.0
                    mujoco.mj_forward(m, ds)
                    initialised = True
                # physics up to the wall clock; each command is applied at the time it was sent
                t_wall = now - t0
                if t_wall - ds.time > 0.5:  # renderer starved (e.g. machine asleep): do not fast-forward
                    ds.time = t_wall - 0.5
                i = 0
                while ds.time < t_wall:
                    while i < len(cmds) and cmds[i][0] - t0 <= ds.time:
                        self._set_ctrl(cmds[i][1])
                        i += 1
                    mujoco.mj_step(m, ds)
                for _, q in cmds[i:]:  # (should not happen) commands newer than the last step
                    self._set_ctrl(q)
                self._set_pose(dr, real if real is not None else target)
                mujoco.mj_forward(m, dr)
            img = self._compose(renderer, cam, ds, dr, t0, now, target, real, label)
            ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            dt = time.monotonic() - last
            last = time.monotonic()
            fps_ema = 0.9 * fps_ema + 0.1 / max(dt, 1e-3)
            with self.lock:
                self.jpeg, self.frame_id = jpeg.tobytes(), self.frame_id + 1
                self.status["fps"] = fps_ema
            time.sleep(max(0.0, period - (time.monotonic() - now)))
        renderer.close()

    def _compose(self, renderer, cam, ds, dr, t0, now, target, real, label):
        panes = []
        for d in (ds, dr):
            renderer.update_scene(d, cam)
            panes.append(renderer.render()[:, :, ::-1])
        img = np.ascontiguousarray(np.hstack(panes))
        W = img.shape[1]
        has_real = real is not None
        right = "REAL (measured)" if has_real else "TARGET (commanded)"
        t = (now - t0) if t0 is not None else 0.0
        qs = {j: float(ds.qpos[self.adr[j]]) for j in JOINTS}
        qr = {j: float(dr.qpos[self.adr[j]]) for j in JOINTS}
        dev = {j: qs[j] - qr[j] for j in JOINTS} if t0 is not None else {j: 0.0 for j in JOINTS}

        # stats + status (under the lock: read by the HTTP thread)
        with self.lock:
            if t0 is not None:
                for j in JOINTS:
                    self.max_dev[j] = max(self.max_dev[j], abs(dev[j]))
                    if has_real and j in target:
                        self.max_lag[j] = max(self.max_lag[j], abs(real[j] - target[j]))
            deg = math.degrees
            self.status.update({
                "t": t, "label": label, "has_real": has_real,
                "max_dev_all": deg(max(self.max_dev.values())),
                "max_lag_all": deg(max(self.max_lag.values())) if has_real else None,
                "joints": [{"name": j,
                            "target": deg(target[j]) if target and j in target else None,
                            "sim": deg(qs[j]), "real": deg(qr[j]) if t0 is not None else None,
                            "dev": deg(dev[j]), "max_dev": deg(self.max_dev[j]),
                            "max_lag": deg(self.max_lag[j]) if has_real else None} for j in JOINTS]})

        font = cv2.FONT_HERSHEY_SIMPLEX
        # top banner: label + time
        cv2.rectangle(img, (0, 0), (W, 38), (20, 20, 20), -1)
        cv2.putText(img, label or "waiting for the first command", (12, 27), font, 0.75, (235, 235, 235), 2,
                    cv2.LINE_AA)
        txt = f"t {t:6.1f} s"
        (tw, _), _ = cv2.getTextSize(txt, font, 0.75, 2)
        cv2.putText(img, txt, (W - tw - 12, 27), font, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
        # pane titles
        for x, name, col in ((0, "SIM (MuJoCo physics)", (255, 200, 80)),
                             (PW, right, (80, 220, 120) if has_real else (80, 200, 255))):
            tw = cv2.getTextSize(name, font, 0.75, 2)[0][0]
            cv2.rectangle(img, (x + 8, 48), (x + 24 + tw, 84), (30, 30, 30), -1)
            cv2.putText(img, name, (x + 16, 74), font, 0.75, col, 2, cv2.LINE_AA)
        cv2.line(img, (PW, 38), (PW, img.shape[0]), (60, 60, 60), 2)
        # bottom strip: SIM - REAL per joint (deg)
        y0 = img.shape[0] - 40
        cv2.rectangle(img, (0, y0), (W, img.shape[0]), (20, 20, 20), -1)
        x = 12
        head = f"SIM - {'REAL' if has_real else 'TARGET'} (deg):"
        cv2.putText(img, head, (x, y0 + 27), font, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
        x += cv2.getTextSize(head, font, 0.65, 2)[0][0] + 18
        for j in JOINTS:
            v = math.degrees(dev[j])
            col = (100, 210, 100) if abs(v) < 3 else (60, 170, 240) if abs(v) < 8 else (80, 80, 240)
            s = f"{SHORT[j]} {v:+5.1f}"
            cv2.putText(img, s, (x, y0 + 27), font, 0.65, col, 2, cv2.LINE_AA)
            x += cv2.getTextSize(s, font, 0.65, 2)[0][0] + 22
        mx = math.degrees(max(self.max_dev.values()))
        s = f"max {mx:4.1f}"
        cv2.putText(img, s, (W - cv2.getTextSize(s, font, 0.65, 2)[0][0] - 12, y0 + 27), font, 0.65,
                    (200, 200, 200), 2, cv2.LINE_AA)
        return img


def _make_handler(twin):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/status.json":
                with twin.lock:
                    body = json.dumps(twin.status).encode()
                self._send(body, "application/json")
            elif path == "/shot.jpg":
                with twin.lock:
                    jpeg = twin.jpeg
                if jpeg is None:
                    self.send_error(503, "no frame yet")
                else:
                    self._send(jpeg, "image/jpeg")
            elif path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = -1
                try:
                    while not twin._stop.is_set():
                        with twin.lock:
                            jpeg, fid = twin.jpeg, twin.frame_id
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

    return Handler
