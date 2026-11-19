"""Synthetic check: render the marker seen by a known camera, recover the camera pose.

    uv run --with opencv-python-headless --with numpy python3 -m unittest discover \
        -s ros2_ws/src/so101_perception/test
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "so101_perception"))
import calibrate_camera_aruco as cca  # noqa: E402

K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
D = np.zeros(5)
SIZE = (1280, 720)


def camera_to_robot(position, target):
    """OpenCV camera (x right, y down, z forward) at `position` looking at `target`."""
    z = np.asarray(target, float) - position
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = position
    return T


def render(T_cam_robot, T_robot_marker):
    """White table with the marker (id 0) drawn, as seen by the camera."""
    cells = 6
    px = 80
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cca.DICTIONARY),
                                           cca.MARKER_ID, cells * px, borderBits=1)
    # outer corners of the marker image (TL, TR, BR, BL) <-> same corners in 3D
    pts3d = cca.marker_corners()
    pts_robot = (T_robot_marker[:3, :3] @ pts3d.T).T + T_robot_marker[:3, 3]
    rvec, _ = cv2.Rodrigues(T_cam_robot[:3, :3])
    pix, _ = cv2.projectPoints(pts_robot, rvec, T_cam_robot[:3, 3], K, D)
    src = np.float32([[0, 0], [cells * px, 0], [cells * px, cells * px], [0, cells * px]])
    H = cv2.getPerspectiveTransform(src, pix.reshape(4, 2).astype(np.float32))
    img = np.full((SIZE[1], SIZE[0]), 255, np.uint8)
    warped = cv2.warpPerspective(marker, H, SIZE, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full_like(marker, 255), H, SIZE)
    img[mask > 0] = warped[mask > 0]
    return img


class ArucoCalibrationTest(unittest.TestCase):
    def test_marker_frame_is_right_handed_and_flat(self):
        for yaw in (0, 90, 37):
            R = cca.marker_to_robot(0.1, 0, yaw)[:3, :3]
            self.assertAlmostEqual(np.linalg.det(R), 1.0)
            np.testing.assert_allclose(R[:, 2], [0, 0, 1])
        up = cca.marker_to_robot(0, 0, 0)[:3, 1]
        np.testing.assert_allclose(up, [1, 0, 0])  # top edge away from the robot
        right = cca.marker_to_robot(0, 0, 0)[:3, 0]
        np.testing.assert_allclose(right, [0, -1, 0])  # robot's right = -y

    def test_recovers_known_camera_pose(self):
        T_rm = cca.marker_to_robot(0.10, 0.0, 0.0)
        T_true = camera_to_robot(np.array([0.45, 0.10, 0.50]), [0.10, 0.0, 0.0])
        img = render(np.linalg.inv(T_true), T_rm)
        corners = cca.detect_marker(img)
        self.assertIsNotNone(corners)
        T, rms, _, _ = cca.estimate_camera_pose(corners, K, D, T_rm)
        self.assertLess(rms, 1.0)
        np.testing.assert_allclose(T[:3, 3], T_true[:3, 3], atol=0.005)
        np.testing.assert_allclose(T[:3, :3], T_true[:3, :3], atol=0.01)

    def test_other_marker_orientation(self):
        T_rm = cca.marker_to_robot(0.12, 0.05, 90.0)
        T_true = camera_to_robot(np.array([0.1, 0.5, 0.45]), [0.12, 0.05, 0.0])
        corners = cca.detect_marker(render(np.linalg.inv(T_true), T_rm))
        T, _, _, _ = cca.estimate_camera_pose(corners, K, D, T_rm)
        np.testing.assert_allclose(T[:3, 3], T_true[:3, 3], atol=0.005)

    def test_wrong_orientation_is_rejected_or_wrong(self):
        # Declaring the marker rotated by 180 degrees puts the camera on the wrong side:
        # it must not silently return the true pose.
        T_rm = cca.marker_to_robot(0.10, 0.0, 0.0)
        T_true = camera_to_robot(np.array([0.45, 0.10, 0.50]), [0.10, 0.0, 0.0])
        corners = cca.detect_marker(render(np.linalg.inv(T_true), T_rm))
        wrong = cca.marker_to_robot(0.10, 0.0, 180.0)
        T, _, _, _ = cca.estimate_camera_pose(corners, K, D, wrong)
        self.assertGreater(np.linalg.norm(T[:3, 3] - T_true[:3, 3]), 0.05)

    def test_average_transforms(self):
        T = camera_to_robot(np.array([0.4, 0.1, 0.5]), [0.1, 0, 0])
        avg = cca.average_transforms([T, T, T])
        np.testing.assert_allclose(avg, T, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
