#!/usr/bin/env python3
"""eval_pick_place.py - evaluate the pick & place pipeline in MuJoCo, without ROS.

Runs the real pipeline code (so101_perception.vision + so101_brain.pick_place +
so101_brain.kinematics) against SO101Sim, with the same rates as the ROS nodes
(driver 50 Hz / 30 Hz, perception 10 Hz, brain 20 Hz). Ground truth
(get_object_position) is only used to SCORE the run, never by the pipeline.

    # from the repo root (host or control container)
    python3 tools/eval_pick_place.py --episodes 20          # pick & place success rate
    python3 tools/eval_pick_place.py --ik-test 50           # IK accuracy (milestone S4: < 1 cm)
    python3 tools/eval_pick_place.py --perception-test 30   # perception error vs ground truth
    python3 tools/eval_pick_place.py --episodes 1 --video out.mp4
    python3 tools/eval_pick_place.py --episodes 3 --demo-video demo.mp4   # polished demo: all
        # episodes in one H.264 video, orbiting 3D view, camera + detection insets, banner
    python3 tools/eval_pick_place.py --episodes 20 --oracle  # ground-truth positions instead
        # of perception: tells perception errors from control errors (debug only)
    python3 tools/eval_pick_place.py --perception-test 50 --detector yolo   # YOLO-seg masks
    python3 tools/eval_pick_place.py --perception-test 50 --lighting 0.4    # dimmed lights
        # --lighting: a light gain (0.4 = 40 %) or "random" (lights + colours, as the dataset)

Needs mujoco, pinocchio (pip: pin), opencv, numpy.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "ros2_ws/src/so101_brain"),
                str(ROOT / "ros2_ws/src/so101_perception")]

import shutil  # noqa: E402
import subprocess  # noqa: E402

import cv2  # noqa: E402
from so101_sim import SO101Sim  # noqa: E402

from so101_brain.kinematics import ArmKinematics  # noqa: E402
from so101_brain.pick_place import Observation, PickPlace, PickPlaceConfig, State  # noqa: E402
from so101_perception import vision  # noqa: E402

SCENE = ROOT / "sim/so101_sim/assets/so101/scene_tek5.xml"
URDF = ROOT / "ros2_ws/src/so101_description/urdf/so101.urdf"
YOLO_WEIGHTS = ROOT / "ros2_ws/src/so101_perception/models/so101_seg.pt"
SEGMENTER = None   # vision.YoloSegmenter with --detector yolo, None = HSV
LIGHTING = None    # --lighting


def set_lighting(sim, seed):
    """Applies --lighting to the scene (no-op by default)."""
    if LIGHTING is None:
        return
    from yolo_dataset import Appearance
    look = getattr(sim, "_appearance", None) or Appearance(sim)
    sim._appearance = look
    rng = np.random.default_rng(10_000 + seed)
    if LIGHTING == "random":
        look.randomize(rng)
    else:
        look.randomize(rng, gain=float(LIGHTING), tint=0.0, colour=0.0)
BRAIN_DT = 0.05          # 20 Hz
PERCEPTION_EVERY = 2     # 10 Hz


def camera_of(sim: SO101Sim) -> vision.Camera:
    """What the driver publishes: MuJoCo K with OpenCV pixel centres, MuJoCo camera pose."""
    K = sim.get_camera_intrinsics().copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return vision.Camera(K, sim.get_camera_extrinsics(), axes="mujoco")


def cube_yaw(sim):
    adr = sim.model.joint("pickup_object_free").qposadr[0]
    w, x, y, z = sim.data.qpos[adr + 3: adr + 7]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def in_box(sim, margin=0.048):
    p, b = sim.get_object_position(), sim.get_drop_box_position()
    return bool(abs(p[0] - b[0]) < margin and abs(p[1] - b[1]) < margin and p[2] < 0.06)


def reset_sim(sim: SO101Sim, seed):
    """Reuse one SO101Sim: same initial state as SO101Sim(seed=seed).connect()."""
    import mujoco
    sim._rng = np.random.default_rng(seed)
    mujoco.mj_resetData(sim.model, sim.data)
    mujoco.mj_forward(sim.model, sim.data)
    sim.randomize_object()


def run_episode(seed, max_time=60.0, verbose=False, video=None, oracle=False, recorder=None, sim=None):
    if sim is None:
        sim = SO101Sim(scene_path=SCENE, seed=seed)
        sim.connect()
    else:
        reset_sim(sim, seed)
    set_lighting(sim, seed)
    cam = camera_of(sim)
    kin = ArmKinematics(URDF, seed=seed)
    logs = []
    brain = PickPlace(kin, PickPlaceConfig(), log=lambda lvl, msg: logs.append(f"[{lvl}] {msg}"))

    def observe():
        obs = sim.get_observation()
        joints = {k.removesuffix(".pos"): v for k, v in obs.items()}
        grip = joints.pop("gripper")
        brain.update_joints(joints, grip)

    writer = None
    if video:
        import mujoco
        overview = mujoco.Renderer(sim.model, 480, 640)
        free_cam = mujoco.MjvCamera()
        free_cam.lookat[:] = [0.12, -0.05, 0.05]
        free_cam.distance, free_cam.azimuth, free_cam.elevation = 0.75, 215, -30
        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 1 / (BRAIN_DT * PERCEPTION_EVERY),
                                 (640 + sim.camera_width * 2, 480))
    observe()
    brain.start(0.0)
    command = None
    now = 0.0
    states = []
    for tick in range(int(max_time / BRAIN_DT)):
        now = tick * BRAIN_DT
        if tick % PERCEPTION_EVERY == 0:
            img = np.ascontiguousarray(sim.render_camera()[:, :, ::-1])
            det, mask = vision.detect_cube(img, cam, SEGMENTER)
            if oracle:
                brain.update_object(Observation(sim.get_object_position(), cube_yaw(sim), now))
            elif det is not None and det.score >= vision.MIN_SCORE:
                brain.update_object(Observation(det.position, det.yaw, now))
            box, _ = vision.detect_drop_box(img, cam, SEGMENTER)
            if box is not None and box.score >= vision.MIN_SCORE:
                brain.update_box(Observation(box.position, box.yaw, now))
            if writer is not None:
                overview.update_scene(sim.data, free_cam)
                big = np.ascontiguousarray(overview.render()[:, :, ::-1])
                cv2.putText(big, f"t={now:4.1f}s  {brain.state.value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                vis = img.copy()
                if det is not None:
                    for corner in cam.project(vision.box_corners(det.position[0], det.position[1], det.yaw,
                                                                 vision.CUBE_SIZE)):
                        cv2.circle(vis, (int(corner[0]), int(corner[1])), 2, (0, 255, 0), -1)
                right = np.vstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
                writer.write(np.hstack([big, cv2.resize(right, (sim.camera_width * 2, 480))]))
        command = brain.step(now) or command
        if recorder is not None:
            recorder.frame(sim, brain, now, img, det, cam)
        if not states or states[-1] != brain.state:
            states.append(brain.state)
        # driver: 50 Hz control, 30 Hz joint states -> ~2 + 2 sim calls per brain tick
        for _ in range(2):
            if command is not None:
                action = {f"{k}.pos": v for k, v in command.joints.items()}
                action["gripper.pos"] = command.gripper
                sim.send_action(action)
            observe()
        if brain.state in (State.DONE, State.IDLE) and brain._mode is None:
            break
    # let the cube settle
    for _ in range(50):
        sim.get_observation()
        if recorder is not None and _ % 2 == 0:
            recorder.frame(sim, brain, now, img, det, cam)
    if writer is not None:
        writer.release()
    ok = in_box(sim)
    if verbose:
        print(f"--- seed {seed}: {'SUCCESS' if ok else 'FAIL'} in {now:.1f}s, states: "
              + " ".join(s.value for s in states))
        for line in logs:
            print("   ", line)
    return ok, now, brain.successes, logs


class DemoRecorder:
    """One polished video for several episodes: orbiting 3D MuJoCo view (1280x720) at the
    brain rate (20 fps = real time), external camera with the fitted cube model and the
    mask as insets, and a banner (episode, time, state, successes)."""

    W, H = 1280, 720
    FPS = 1 / BRAIN_DT

    def __init__(self, path, episodes):
        self.path = path
        self.episodes = episodes
        self.raw = str(Path(path).with_suffix(".raw.mp4"))
        self.writer = cv2.VideoWriter(self.raw, cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (self.W, self.H))
        self.episode = 0
        self.successes = 0
        self.frames = 0
        self._renderer = None

    def new_episode(self, seed):
        self.episode += 1
        self.seed = seed

    def end_episode(self, ok):
        self.successes += int(ok)
        # hold the last frame 1 s with the result
        for _ in range(int(self.FPS)):
            self.writer.write(self._last_with(("CUBE IN THE BOX" if ok else "FAILED"),
                                              (80, 220, 80) if ok else (60, 60, 230)))

    def _last_with(self, text, color):
        f = self._last.copy()
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)
        cv2.putText(f, text, ((self.W - tw) // 2, self.H // 2), cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 3,
                    cv2.LINE_AA)
        return f

    def frame(self, sim, brain, now, img, det, cam):
        import mujoco
        if self._renderer is None:  # one SO101Sim for the whole demo -> one renderer
            sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, self.W)
            sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, self.H)
            self._renderer = mujoco.Renderer(sim.model, self.H, self.W)
        cam3d = mujoco.MjvCamera()
        cam3d.lookat[:] = [0.13, -0.06, 0.06]
        cam3d.distance, cam3d.elevation = 0.78, -28
        cam3d.azimuth = 200 + 4.0 * (self.frames / self.FPS)  # slow orbit, 4 deg/s
        self._renderer.update_scene(sim.data, cam3d)
        frame = np.ascontiguousarray(self._renderer.render()[:, :, ::-1])

        # inset 1: external camera + fitted cube model
        vis = img.copy()
        if det is not None:
            ok = det.score >= vision.MIN_SCORE
            pts = cam.project(vision.box_corners(det.position[0], det.position[1], det.yaw, vision.CUBE_SIZE))
            cv2.polylines(vis, [cv2.convexHull(pts.astype(np.int32))], True, (0, 255, 0) if ok else (0, 0, 255), 1)
        inset = cv2.resize(vis, (400, 300))
        x0, y0 = self.W - 400 - 16, 16
        frame[y0:y0 + 300, x0:x0 + 400] = inset
        cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + 400, y0 + 300), (255, 255, 255), 1)
        cv2.putText(frame, "external camera + fitted model", (x0 + 6, y0 + 292), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 150), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        put = lambda t, y, sc, col=(255, 255, 255), th=1: cv2.putText(  # noqa: E731
            frame, t, (16, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
        put("SO-ARM101 pick & place  (MuJoCo + ROS2 pipeline code)", 32, 0.62, th=1)
        put(f"episode {self.episode}/{self.episodes}   t = {now:4.1f} s   placed: {self.successes}", 66, 0.62)
        put(brain.state.value, 108, 1.1, (80, 220, 255), 2)
        put(brain.status[:60], 138, 0.5, (210, 210, 210))
        self._last = frame
        self.writer.write(frame)
        self.frames += 1

    def close(self):
        self.writer.release()
        if self._renderer is not None:
            self._renderer.close()
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", self.raw, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart", self.path], check=True)
            Path(self.raw).unlink()
        else:
            Path(self.raw).rename(self.path)  # mp4v fallback
        print(f"demo video: {self.path} ({self.frames / self.FPS:.0f} s)")


def ik_test(n, seed=0):
    """Reach random XYZ targets in sim with IK; measure the real TCP position in MuJoCo."""
    rng = np.random.default_rng(seed)
    sim = SO101Sim(scene_path=SCENE, seed=seed)
    sim.connect()
    kin = ArmKinematics(URDF, seed=seed)
    errs = []
    q = kin.neutral()
    for i in range(n):
        r, a, z = rng.uniform(0.12, 0.32), rng.uniform(-1.2, 1.2), rng.uniform(0.02, 0.20)
        target = np.array([r * np.cos(a), r * np.sin(a), z])
        res = kin.solve_global(target, q, frame="tcp")
        if not res.ok(1e-3):
            print(f"  {i}: target {np.round(target, 3)} unreachable by IK ({res.position_error * 1000:.1f} mm)")
            continue
        q = res.q
        action = {f"{k}.pos": v for k, v in kin.joints_from_q(q).items()}
        for _ in range(150):
            sim.send_action(action)
        measured = sim.data.site("gripperframe").xpos.copy()
        errs.append(np.linalg.norm(measured - target) * 1000)
    errs = np.array(errs)
    print(f"IK test: {len(errs)}/{n} targets reached, error measured in MuJoCo: "
          f"mean {errs.mean():.2f} mm, max {errs.max():.2f} mm, <10 mm: {np.mean(errs < 10) * 100:.0f}%")


def perception_test(n):
    home = {f"{k}.pos": v for k, v in PickPlaceConfig().home.items()}
    pe, ye, be, t = [], [], [], []
    for seed in range(n):
        sim = SO101Sim(scene_path=SCENE, seed=seed)
        sim.connect()
        set_lighting(sim, seed)
        for _ in range(150):
            sim.send_action(home)
        img = np.ascontiguousarray(sim.render_camera()[:, :, ::-1])
        t0 = time.perf_counter()
        det, _ = vision.detect_cube(img, camera_of(sim), SEGMENTER)
        box, _ = vision.detect_drop_box(img, camera_of(sim), SEGMENTER)
        t.append(time.perf_counter() - t0)
        if box is not None and box.score >= vision.MIN_SCORE:
            be.append(np.linalg.norm((box.position - sim.get_drop_box_position())[:2]) * 1000)
        if det is None or det.score < vision.MIN_SCORE:
            print(f"  seed {seed}: cube not detected" + ("" if det is None else f" (score {det.score:.2f})"))
            continue
        err = np.linalg.norm((det.position - sim.get_object_position())[:2]) * 1000
        dy = (det.yaw - cube_yaw(sim) + np.pi / 4) % (np.pi / 2) - np.pi / 4
        pe.append(err)
        ye.append(abs(np.degrees(dy)))
    name = "yolo" if SEGMENTER is not None else "hsv"
    if not pe:
        print(f"Perception test [{name}, lighting {LIGHTING or 'nominal'}]: cube detected 0/{n}")
        return
    print(f"Perception test [{name}, lighting {LIGHTING or 'nominal'}]: cube detected {len(pe)}/{n}, "
          f"position error mean {np.mean(pe):.1f} mm, max {np.max(pe):.1f} mm; yaw error mean "
          f"{np.mean(ye):.1f} deg, max {np.max(ye):.1f} deg; box detected {len(be)}/{n}"
          + (f", error mean {np.mean(be):.1f} mm" if be else "")
          + f"; {np.median(t) * 1000:.0f} ms/image (cube + box)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--ik-test", type=int, default=0)
    parser.add_argument("--perception-test", type=int, default=0)
    parser.add_argument("--video", help="mp4 of the camera + mask (first episode)")
    parser.add_argument("--demo-video", help="polished H.264 demo of all the episodes in one file")
    parser.add_argument("--seeds", help="comma-separated seeds instead of --first-seed/--episodes")
    parser.add_argument("--oracle", action="store_true", help="ground-truth cube pose instead of perception")
    parser.add_argument("--detector", choices=("hsv", "yolo"), default="hsv", help="segmenter of the perception")
    parser.add_argument("--weights", default=str(YOLO_WEIGHTS), help="YOLO-seg weights (--detector yolo)")
    parser.add_argument("--lighting", help='light gain (e.g. 0.4) or "random": robustness tests')
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    global SEGMENTER, LIGHTING
    LIGHTING = args.lighting
    if args.detector == "yolo":
        SEGMENTER = vision.YoloSegmenter(args.weights, device="cpu")
    if not (args.episodes or args.seeds or args.ik_test or args.perception_test):
        args.episodes = 10

    if args.ik_test:
        ik_test(args.ik_test)
    if args.perception_test:
        perception_test(args.perception_test)
    if args.episodes or args.seeds:
        results = []
        t0 = time.perf_counter()
        seeds = ([int(v) for v in args.seeds.split(",")] if args.seeds
                 else [args.first_seed + i for i in range(args.episodes)])
        recorder = DemoRecorder(args.demo_video, len(seeds)) if args.demo_video else None
        shared_sim = None
        if recorder is not None:  # one simulation, reset per episode (a single OpenGL context)
            shared_sim = SO101Sim(scene_path=SCENE, seed=seeds[0])
            shared_sim.connect()
        for i, seed in enumerate(seeds):
            if recorder is not None:
                recorder.new_episode(seed)
            ok, dur, _, logs = run_episode(seed, verbose=args.verbose, video=args.video if i == 0 else None,
                                          oracle=args.oracle, recorder=recorder, sim=shared_sim)
            if recorder is not None:
                recorder.end_episode(ok)
            results.append((ok, dur))
            if not args.verbose:
                print(f"seed {seed:3d}: {'OK  ' if ok else 'FAIL'} {dur:5.1f}s "
                      + ("" if ok else "| " + " | ".join(line for line in logs if "[info]" not in line)))
        if recorder is not None:
            recorder.close()
        oks = [r[0] for r in results]
        durs = [r[1] for r in results if r[0]]
        print(f"\nPick & place: {sum(oks)}/{len(oks)} successful ({100 * np.mean(oks):.0f}%), "
              f"mean time {np.mean(durs) if durs else float('nan'):.1f}s (sim), "
              f"wall {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
