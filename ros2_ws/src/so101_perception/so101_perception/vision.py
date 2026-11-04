"""vision.py - pure image processing for the pick & place (no ROS).

Used by perception_node and by tools/eval_pick_place.py.

Pipeline for each object:
  1. HSV threshold -> binary mask (largest blob, holes filled).
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


def detect_box_object(bgr, camera: Camera, ranges, size, z0=0.0, min_area=MIN_BLOB_AREA_PX,
                      dilate_px: int = 0):
    """Detect a box-shaped object of known size. Returns Detection or None, and the mask."""
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


def detect_cube(bgr, camera: Camera):
    return detect_box_object(bgr, camera, CUBE_HSV_RANGES, CUBE_SIZE, dilate_px=CUBE_DILATE_PX)


def detect_drop_box(bgr, camera: Camera):
    return detect_box_object(bgr, camera, BOX_HSV_RANGES, BOX_SIZE)
