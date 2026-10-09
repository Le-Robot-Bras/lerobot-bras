"""Offline chessboard intrinsics and measured robot-frame extrinsics."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def estimate_camera_pose(world_points, image_points, K, D):
    """Return OpenCV camera->robot transform and reprojection RMS in pixels."""
    world = np.asarray(world_points, dtype=float).reshape(-1, 3)
    pixels = np.asarray(image_points, dtype=float).reshape(-1, 2)
    if len(world) < 6 or len(world) != len(pixels):
        raise ValueError("Provide at least six corresponding world/image points")
    if not np.all(np.isfinite(world)) or not np.all(np.isfinite(pixels)):
        raise ValueError("Reference points must be finite")
    if np.linalg.matrix_rank(world - world.mean(axis=0)) < 2:
        raise ValueError("Reference points must not be collinear")
    ok, rvec, tvec = cv2.solvePnP(world, pixels, K, D)
    if not ok:
        raise ValueError("Camera pose estimation failed")
    R, _ = cv2.Rodrigues(rvec)
    if np.any((world @ R.T + tvec.reshape(3))[:, 2] <= 0):
        raise ValueError("Reference points lie behind the estimated camera")
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ tvec.reshape(3)
    projected, _ = cv2.projectPoints(world, rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1))))
    return T, rms


def calibrate_intrinsics(images, cols, rows, square):
    """Calibrate raw camera images with internal chessboard corner counts."""
    if cols < 2 or rows < 2 or not np.isfinite(square) or square <= 0:
        raise ValueError("Invalid chessboard dimensions or square size")
    grid = np.zeros((cols * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    objects, pixels, size = [], [], None
    for filename in images:
        image = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read {filename}")
        current_size = (image.shape[1], image.shape[0])
        if size is not None and current_size != size:
            raise ValueError("All calibration images must have the same resolution")
        size = current_size
        found, corners = cv2.findChessboardCornersSB(image, (cols, rows))
        if found:
            objects.append(grid.copy())
            pixels.append(corners)
    if len(objects) < 10:
        raise ValueError("Need at least ten images with a detected chessboard")
    rms, K, D, _, _ = cv2.calibrateCamera(objects, pixels, size, None, None)
    return K, D, size, float(rms), len(objects)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--cols', type=int, required=True, help='Internal corner columns')
    parser.add_argument('--rows', type=int, required=True, help='Internal corner rows')
    parser.add_argument('--square', type=float, required=True, help='Square side in metres')
    parser.add_argument('--reference', required=True, help='JSON world_points and image_points')
    parser.add_argument('--output', required=True)
    parser.add_argument('--frame-id', default='world')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; choose another filename')
    try:
        K, D, size, rms, count = calibrate_intrinsics(args.images, args.cols, args.rows, args.square)
        reference = json.loads(Path(args.reference).read_text())
        T, pose_rms = estimate_camera_pose(reference['world_points'], reference['image_points'], K, D)
        data = dict(K=K.tolist(), D=D.ravel().tolist(), T=T.tolist(),
                    image_size=list(size), camera_axes='opencv', frame_id=args.frame_id,
                    intrinsic_rms_px=rms, pose_rms_px=pose_rms, calibration_images=count)
        output.write_text(json.dumps(data, indent=2) + '\n')
    except (ValueError, KeyError, OSError, cv2.error) as exc:
        parser.error(str(exc))
    print(f'Saved {output}: intrinsic RMS {rms:.3f} px, pose RMS {pose_rms:.3f} px')


if __name__ == '__main__':
    main()
