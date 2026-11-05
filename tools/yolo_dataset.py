#!/usr/bin/env python3
"""yolo_dataset.py - synthetic YOLO-seg dataset of the external camera, labelled by MuJoCo.

Every image is rendered by the simulated external camera (320x240), and the labels come
from MuJoCo's segmentation rendering, so they are exact and free. Two classes:
  0 cube      (pickup_object, tunnels filled: we want its silhouette)
  1 drop_box

Domain randomization, so that the network learns the SHAPE and not one exact colour
(this is what the HSV thresholds cannot do):
  - lights: intensity x0.25 to x1.6, colour tint, direction
  - cube / box / table colours: hue, saturation and brightness jitter
  - cube: random position and yaw (same distribution as the sim), arm: random pose
    (partial occlusions, as when the gripper is above the cube), box: moved on the table

    python3 tools/yolo_dataset.py --out datasets/so101_seg --images 6000
    # then: python3 tools/yolo_train.py --data datasets/so101_seg/data.yaml

Needs mujoco, opencv, numpy.
"""
import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
from so101_sim import SO101Sim  # noqa: E402

SCENE = ROOT / "sim/so101_sim/assets/so101/scene_tek5.xml"
CLASSES = ("cube", "drop_box")
MIN_AREA_PX = 20
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


class Appearance:
    """Randomizes lights and colours of a SO101Sim, and restores them."""

    def __init__(self, sim: SO101Sim):
        m = sim.model
        self.m = m
        self.saved = {
            "light_diffuse": m.light_diffuse.copy(), "light_ambient": m.light_ambient.copy(),
            "light_dir": m.light_dir.copy(), "mat_rgba": m.mat_rgba.copy(),
            "head_diffuse": m.vis.headlight.diffuse.copy(), "head_ambient": m.vis.headlight.ambient.copy(),
        }
        self.materials = [m.material(n).id for n in ("red", "box_mat", "groundplane")]

    def restore(self):
        m, s = self.m, self.saved
        m.light_diffuse[:], m.light_ambient[:], m.light_dir[:] = s["light_diffuse"], s["light_ambient"], s["light_dir"]
        m.mat_rgba[:] = s["mat_rgba"]
        m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:] = s["head_diffuse"], s["head_ambient"]

    def randomize(self, rng, gain=None, tint=0.15, colour=1.0):
        """gain: global light factor (None = random in [0.25, 1.6])."""
        self.restore()
        m, s = self.m, self.saved
        gain = rng.uniform(0.25, 1.6) if gain is None else gain
        col = 1 + rng.uniform(-tint, tint, 3)
        m.light_diffuse[:] = np.clip(s["light_diffuse"] * gain * col, 0, 1)
        m.light_ambient[:] = np.clip(s["light_ambient"] * gain * col, 0, 1)
        m.vis.headlight.diffuse[:] = np.clip(s["head_diffuse"] * gain * col, 0, 1)
        m.vis.headlight.ambient[:] = np.clip(s["head_ambient"] * gain * col, 0, 1)
        d = s["light_dir"] + rng.normal(0, 0.25, s["light_dir"].shape)
        m.light_dir[:] = d / np.linalg.norm(d, axis=1, keepdims=True)
        for mid in self.materials:
            rgb = s["mat_rgba"][mid, :3]
            hsv = cv2.cvtColor(np.uint8([[np.clip(rgb, 0, 1) * 255]]), cv2.COLOR_RGB2HSV)[0, 0].astype(float)
            hsv[0] = (hsv[0] + rng.uniform(-8, 8) * colour) % 180
            hsv[1] = np.clip(hsv[1] * (1 + rng.uniform(-0.35, 0.2) * colour), 0, 255)
            hsv[2] = np.clip(hsv[2] * (1 + rng.uniform(-0.3, 0.3) * colour), 0, 255)
            m.mat_rgba[mid, :3] = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0] / 255.0


def class_geoms(m):
    cube = {m.geom("pickup_object_visual").id}
    box_body = m.body("drop_box").id
    box = {g for g in range(m.ngeom) if m.geom_bodyid[g] == box_body}
    return cube, box


def random_arm(sim, rng):
    """Random arm pose: half of the time hovering near the cube (occlusions)."""
    m, d = sim.model, sim.data
    for name in ARM:
        lo, hi = m.jnt_range[m.joint(name).id]
        d.qpos[sim._joint_qpos_adr[name]] = rng.uniform(lo, hi) * rng.uniform(0.2, 0.8)
    if rng.random() < 0.5:  # point roughly at the cube, arm lowered
        p = sim.get_object_position()
        d.qpos[sim._joint_qpos_adr["shoulder_pan"]] = np.arctan2(p[1], p[0]) + rng.normal(0, 0.15)
        d.qpos[sim._joint_qpos_adr["shoulder_lift"]] = rng.uniform(-0.3, 0.9)
        d.qpos[sim._joint_qpos_adr["elbow_flex"]] = rng.uniform(-0.6, 0.9)
        d.qpos[sim._joint_qpos_adr["wrist_flex"]] = rng.uniform(0.3, 1.6)
    d.qpos[sim._joint_qpos_adr["gripper"]] = rng.uniform(0, 1.5)


def move_box(sim, rng, home):
    m = sim.model
    bid = m.body("drop_box").id
    m.body_pos[bid, :2] = home[:2] + rng.uniform(-0.04, 0.04, 2)
    yaw = rng.uniform(-0.4, 0.4)
    m.body_quat[bid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def polygons(mask):
    """YOLO-seg polygon of the largest outer contour (holes ignored), or None."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA_PX or len(c) < 3:
        return None
    return c.reshape(-1, 2)


def render_pair(sim, rgb_ren, seg_ren):
    rgb_ren.update_scene(sim.data, camera=sim.camera_name)
    seg_ren.update_scene(sim.data, camera=sim.camera_name)
    return rgb_ren.render()[:, :, ::-1].copy(), seg_ren.render()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "datasets/so101_seg"))
    ap.add_argument("--images", type=int, default=6000)
    ap.add_argument("--val", type=float, default=0.15, help="validation fraction")
    ap.add_argument("--seed", type=int, default=12345, help="far from the eval seeds (0-199)")
    args = ap.parse_args()

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sim = SO101Sim(scene_path=SCENE, seed=args.seed)
    sim.connect()
    m, d = sim.model, sim.data
    look = Appearance(sim)
    cube_g, box_g = class_geoms(m)
    box_home = m.body_pos[m.body("drop_box").id].copy()
    rgb_ren = mujoco.Renderer(m, sim.camera_height, sim.camera_width)
    seg_ren = mujoco.Renderer(m, sim.camera_height, sim.camera_width)
    seg_ren.enable_segmentation_rendering()
    H, W = sim.camera_height, sim.camera_width
    counts = np.zeros(2, int)
    for i in range(args.images):
        mujoco.mj_resetData(m, d)
        move_box(sim, rng, box_home)
        sim._rng = np.random.default_rng(rng.integers(1 << 31))
        sim.randomize_object()          # settles the cube with physics
        random_arm(sim, rng)
        mujoco.mj_forward(m, d)
        look.randomize(rng)
        img, seg = render_pair(sim, rgb_ren, seg_ren)
        geom = np.where(seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[:, :, 0], -1)
        lines = []
        for cls, ids in enumerate((cube_g, box_g)):
            poly = polygons(np.isin(geom, list(ids)))
            if poly is not None:
                counts[cls] += 1
                xy = (poly / [W, H]).clip(0, 1).ravel()
                lines.append(f"{cls} " + " ".join(f"{v:.5f}" for v in xy))
        split = "val" if i < args.images * args.val else "train"
        cv2.imwrite(str(out / "images" / split / f"{i:05d}.png"), img)
        (out / "labels" / split / f"{i:05d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if i % 200 == 0:
            print(f"{i}/{args.images}", flush=True)
    look.restore()
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        + "names:\n" + "".join(f"  {k}: {n}\n" for k, n in enumerate(CLASSES)))
    print(f"done: {args.images} images, cube visible in {counts[0]}, box in {counts[1]} -> {out}/data.yaml")


if __name__ == "__main__":
    main()
