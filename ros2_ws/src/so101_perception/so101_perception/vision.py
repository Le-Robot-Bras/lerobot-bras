"""vision.py - pure image processing for the pick & place (no ROS).

Used by perception_node and by tools/eval_pick_place.py.

Pipeline for each object:
  1. Segmentation -> binary mask (largest blob, holes filled). Two segmenters:
     - "hsv" (default): colour thresholds. No training, ~1 ms, but fixed colours.
     - "yolo": YOLO-seg network (YoloSegmenter) trained on synthetic MuJoCo images
       with randomized lights and colours (tools/yolo_dataset.py + yolo_train.py),
       then refined at the pixel (refine_mask): the colour of the object is learnt
       in the current frame inside the YOLO mask. YOLO finds the object whatever the
       light, the refinement gives back the precision of the thresholds.
     Steps 2-3 are the same for both: the network only replaces the mask.
  2. First guess: back-project the blob centroid onto the plane z = h/2.
  3. Model fit: the object is a known box (w x d x h) lying on the table. Search
     (x, y, yaw) so that the projection of its 8 corners (convex hull, clipped
     to the image like the real view) best overlaps the mask (IoU).
     This removes the centroid bias of a perspective view (side faces), works
     when the object is cut by the image border, and gives the yaw for free.

Camera model: K, distortion D, T = camera -> robot transform (4x4). `axes` says
which camera convention T uses: "opencv" (x right, y down, z forward, e.g. from
calibrate_camera.py) or "mujoco" (x right, y up, z backward, from the sim).
"""
from dataclasses import dataclass

import cv2
import numpy as np

# Red cube. Measured on the MuJoCo scene (OpenCV H in 0-179): the cube is H ~168.
# Real red can also wrap around H=0 (high saturation only, to exclude brown).
# S/V minimums low enough to keep the shaded faces and edges of the cube.
CUBE_HSV_RANGES = (((155, 100, 50), (180, 255, 255)), ((0, 190, 50), (8, 255, 255)))
# The thresholded silhouette of the cube is ~1 px smaller than the true one all around
# (anti-aliased, darker rounded edges): dilate it by 1 px before the model fit.
# Measured in sim: cube error 2.1 -> 0.8 mm mean, 5.3 -> 1.7 mm max. Not for the box
# (thin walls, its error gets worse).
CUBE_DILATE_PX = 1
# YOLO + refine_mask: same 1 px shrink as the thresholds (colour classification at the
# anti-aliased edges). Measured in sim (40 scenes): 0.9 mm mean / 2.3 mm max, 2.5 deg;
# at 20 % light 0.8 / 1.7 mm (HSV: cube lost). Raw YOLO masks: 2.8 / 8.2 mm, 6 deg.
YOLO_CUBE_DILATE_PX = 1
# Brown drop box.
BOX_HSV_RANGES = (((5, 80, 25), (25, 230, 220)),)

CUBE_SIZE = (0.05, 0.05, 0.05)          # m
BOX_SIZE = (0.104, 0.104, 0.05)         # outer walls (m)

MIN_BLOB_AREA_PX = 30
# Below this IoU between the fitted model and the mask, the detection is not trusted
# (object partly hidden by the arm, wrong blob...).
MIN_SCORE = 0.75
MUJOCO_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


@dataclass
class Detection:
    position: np.ndarray   # centre of the object (m, robot frame)
    yaw: float             # rad, modulo the object symmetry (pi/2 for a square)
    score: float           # IoU of the fitted model with the mask (0-1)
    pixel: tuple           # blob centroid (u, v)
    touches_border: bool   # blob cut by the image border (less reliable)


class Camera:
    """Pinhole camera with its pose in the robot frame."""

    def __init__(self, K, T, D=None, axes: str = "opencv", size=None):
        self.K = np.asarray(K, dtype=float).reshape(3, 3)
        T = np.asarray(T, dtype=float).reshape(4, 4)
        if axes not in ("opencv", "mujoco"):
            raise ValueError("axes must be 'opencv' or 'mujoco'")
        # Store the OpenCV-convention camera -> robot transform.
        self.T = T.copy()
        if axes == "mujoco":
            self.T[:3, :3] = T[:3, :3] @ MUJOCO_TO_OPENCV
        self.D = np.zeros(5) if D is None or len(D) == 0 else np.asarray(D, dtype=float)
        self.size = size  # (width, height) or None
        # robot -> camera, for projectPoints
        R_wc = self.T[:3, :3].T
        self._rvec, _ = cv2.Rodrigues(R_wc)
        self._tvec = -R_wc @ self.T[:3, 3]

    @property
    def position(self) -> np.ndarray:
        return self.T[:3, 3]

    def ray_to_plane(self, u, v, z):
        """Back-project pixel (u, v) onto the horizontal plane z. None if parallel / behind."""
        x, y = cv2.undistortPoints(np.array([[[u, v]]], dtype=float), self.K, self.D).reshape(2)
        direction = self.T[:3, :3] @ np.array([x, y, 1.0])
        origin = self.T[:3, 3]
        if abs(direction[2]) < 1e-9:
            return None
        t = (z - origin[2]) / direction[2]
        if t <= 0:
            return None
        return origin + t * direction

    def project(self, points) -> np.ndarray:
        pts, _ = cv2.projectPoints(np.asarray(points, dtype=float).reshape(-1, 1, 3),
                                   self._rvec, self._tvec, self.K, self.D)
        return pts.reshape(-1, 2)


def hsv_mask(bgr, ranges, close_px: int = 5) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    if close_px > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    return mask


def largest_blob(mask, min_area=MIN_BLOB_AREA_PX):
    """Filled silhouette of the largest blob (holes such as the cube tunnels closed)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    blob = max(contours, key=cv2.contourArea)
    if cv2.contourArea(blob) < min_area:
        return None, None
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [blob], -1, 255, cv2.FILLED)
    return blob, filled


def box_corners(x, y, yaw, size, z0=0.0) -> np.ndarray:
    w, d, h = size
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[sx * w / 2, sy * d / 2] for sx in (-1, 1) for sy in (-1, 1)])
    xy = local @ np.array([[c, s], [-s, c]]) + [x, y]
    return np.array([[px, py, z0 + pz] for pz in (0.0, h) for px, py in xy])


class BoxFitter:
    """Fit (x, y, yaw) of a box of known size lying on the table to a silhouette."""

    def __init__(self, camera: Camera, size, z0: float = 0.0, symmetry: float = np.pi / 2):
        self.camera = camera
        self.size = size
        self.z0 = z0
        self.symmetry = symmetry

    def silhouette(self, x, y, yaw, shape) -> np.ndarray:
        """Filled projection of the box. Drawn in an image of the camera size, so it
        is clipped by the border exactly like the real view."""
        pts = self.camera.project(box_corners(x, y, yaw, self.size, self.z0))
        hull = cv2.convexHull(np.round(pts * 4).astype(np.int32))  # 2 bits of sub-pixel precision
        sil = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(sil, hull, 255, lineType=cv2.LINE_8, shift=2)
        return sil

    def iou(self, params, target) -> float:
        sil = self.silhouette(*params, target.shape)
        inter = np.count_nonzero(sil & target)
        union = np.count_nonzero(sil | target)
        return inter / union if union else 0.0

    def fit(self, target: np.ndarray, guess_xy, yaw_seeds: int = 3):
        """Pattern search on (x, y, yaw) from a few yaw seeds. Returns (x, y, yaw, iou)."""
        best = None
        for yaw0 in np.arange(yaw_seeds) * self.symmetry / yaw_seeds:
            p = np.array([guess_xy[0], guess_xy[1], yaw0])
            score = self.iou(p, target)
            step = np.array([0.004, 0.004, np.radians(10)])
            while step[0] > 2.5e-4:
                improved = False
                for i in range(3):
                    for sign in (1, -1):
                        cand = p.copy()
                        cand[i] += sign * step[i]
                        s = self.iou(cand, target)
                        if s > score:
                            p, score, improved = cand, s, True
                if not improved:
                    step /= 2
            if best is None or score > best[1]:
                best = (p, score)
        p, score = best
        return p[0], p[1], float(np.mod(p[2], self.symmetry)), score


def refine_mask(bgr, mask, ring: int = 4) -> np.ndarray:
    """Pixel-accurate silhouette from a coarse mask (YOLO protos are 1/4 resolution).

    Two colour models learnt in THIS frame (Lab, mean and std): the object = the eroded
    coarse mask, the background = a ring around it. Each pixel near the mask goes to
    the closest model (normalised distance). Works under any light, since the colours
    are measured, not fixed in advance."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    core = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
    zone = cv2.dilate(mask, np.ones((2 * ring + 1, 2 * ring + 1), np.uint8)) > 0
    bg = zone & ~(cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0)
    if core.sum() < 10 or bg.sum() < 10:
        return mask
    d = []
    for sel in (core, bg):
        mean, std = lab[sel].mean(0), lab[sel].std(0) + 3.0
        d.append((((lab - mean) / std) ** 2).sum(-1))
    refined = ((d[0] < d[1]) & zone).astype(np.uint8) * 255
    _, filled = largest_blob(refined)
    return mask if filled is None else filled


class YoloSegmenter:
    """Masks of the cube and the drop box from a YOLO segmentation model (ultralytics).

    One forward pass per image for both classes; keeps the most confident instance of
    each class. Masks are rebuilt from the predicted polygons at the image resolution."""

    CLASSES = ("cube", "drop_box")

    def __init__(self, weights, conf: float = 0.25, device=None, refine: bool = True):
        from ultralytics import YOLO  # optional dependency (AGPL-3.0), only for "yolo"
        self.model = YOLO(str(weights), task="segment")
        self.conf = conf
        self.device = device
        self.refine = refine
        self._cache = (None, None)

    def masks(self, bgr) -> dict:
        """{class name: uint8 mask (0/255)} for the classes found in the image."""
        if self._cache[0] is bgr:  # detect_cube + detect_drop_box on the same frame
            return self._cache[1]
        res = self.model.predict(bgr, imgsz=max(bgr.shape[:2]), conf=self.conf, device=self.device,
                                 verbose=False)[0]
        out, best = {}, {}
        if res.masks is not None:
            for poly, cls, conf in zip(res.masks.xy, res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                name = self.CLASSES[int(cls)] if int(cls) < len(self.CLASSES) else None
                if name is None or len(poly) < 3 or conf <= best.get(name, 0.0):
                    continue
                mask = np.zeros(bgr.shape[:2], np.uint8)
                cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
                out[name], best[name] = mask, conf
        if self.refine:
            out = {name: refine_mask(bgr, mask) for name, mask in out.items()}
        self._cache = (bgr, out)
        return out


def detect_box_object(bgr, camera: Camera, ranges, size, z0=0.0, min_area=MIN_BLOB_AREA_PX,
                      dilate_px: int = 0, mask=None):
    """Detect a box-shaped object of known size. Returns Detection or None, and the mask.
    `mask`: segmentation given by another segmenter (YOLO); default: HSV thresholds."""
    if mask is None:
        mask = hsv_mask(bgr, ranges)
    if dilate_px > 0:
        k = 2 * dilate_px + 1  # symmetric kernel: a 2x2 kernel would shift the silhouette
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    blob, filled = largest_blob(mask, min_area)
    if blob is None:
        return None, mask
    m = cv2.moments(blob)
    u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
    guess = camera.ray_to_plane(u, v, z0 + size[2] / 2)
    if guess is None:
        return None, mask
    x, y, yaw, score = BoxFitter(camera, size, z0).fit(filled, guess[:2])
    bx, by, bw, bh = cv2.boundingRect(blob)
    hgt, wid = mask.shape
    border = bx <= 0 or by <= 0 or bx + bw >= wid or by + bh >= hgt
    det = Detection(np.array([x, y, z0 + size[2] / 2]), yaw, score, (u, v), border)
    return det, filled


def _segmented(bgr, segmenter, name):
    if segmenter is None:
        return None
    return segmenter.masks(bgr).get(name, np.zeros(bgr.shape[:2], np.uint8))


def detect_cube(bgr, camera: Camera, segmenter: YoloSegmenter | None = None):
    mask = _segmented(bgr, segmenter, "cube")
    dilate = CUBE_DILATE_PX if segmenter is None else YOLO_CUBE_DILATE_PX
    return detect_box_object(bgr, camera, CUBE_HSV_RANGES, CUBE_SIZE, dilate_px=dilate, mask=mask)


def detect_drop_box(bgr, camera: Camera, segmenter: YoloSegmenter | None = None):
    return detect_box_object(bgr, camera, BOX_HSV_RANGES, BOX_SIZE, mask=_segmented(bgr, segmenter, "drop_box"))
