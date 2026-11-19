"""Camera extrinsics (camera -> robot) from ONE ArUco marker lying flat at a known place.

Marker: DICT_4X4_1000, id 0, 100 mm side (outer edge of the black border), flat on the table.
Its position in the robot frame (base_link, metres, x forward, y left, z up) is given on the
command line. Intrinsics (K, D) come either from an existing calibration JSON or from
chessboard images (same procedure as calibrate_camera.py).

    # intrinsics already known (JSON with K, D, image_size), picture(s) of the marker
    python3 calibrate_camera_aruco.py --intrinsics cam_intrinsics.json \\
        --image shot1.png shot2.png --marker-x 0.10 --output camera.json

    # live camera, 20 frames averaged
    python3 calibrate_camera_aruco.py --intrinsics cam_intrinsics.json --device 0 \\
        --width 1280 --height 720 --marker-x 0.10 --output camera.json

The output has the same format as calibrate_camera.py (K, D, T, ...): give it to the
perception through CAMERA_CALIBRATION in docker/.env. A debug image with the detected marker
and its axes is written next to it (<output>.debug.png): CHECK IT.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from so101_perception.calibrate_camera import calibrate_intrinsics
except ImportError:  # run as a plain script
    from calibrate_camera import calibrate_intrinsics

DICTIONARY = cv2.aruco.DICT_4X4_1000
MARKER_ID = 0
MARKER_SIDE = 0.100  # m, outer edge of the black border


def marker_corners(side=MARKER_SIDE):
    """Corners in the marker frame (centre origin, x right, y up, z out of the paper).

    Order is the ArUco one: top-left, top-right, bottom-right, bottom-left.
    """
    h = side / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def marker_to_robot(x, y, yaw_deg=0.0, z=0.0):
    """4x4 pose of the marker frame in the robot frame, marker lying flat, face up.

    (x, y, z) is the CENTRE of the marker. yaw_deg = direction in which the marker's top
    edge points, counter-clockwise seen from above: 0 -> towards +x (away from the robot,
    the marker reads upright from the robot's side), 90 -> towards +y (the robot's left).
    """
    a = np.radians(yaw_deg)
    up = np.array([np.cos(a), np.sin(a), 0.0])  # marker +y
    right = np.array([np.sin(a), -np.cos(a), 0.0])  # marker +x (up turned -90 deg)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([right, up, [0.0, 0.0, 1.0]])
    T[:3, 3] = [x, y, z]
    return T


def detect_marker(image, marker_id=MARKER_ID):
    """Pixel corners (4, 2) of the marker, or None."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    for c, i in zip(corners, ids.ravel()):
        if int(i) == marker_id:
            return c.reshape(4, 2).astype(np.float64)
    return None


def estimate_camera_pose(pixels, K, D, T_robot_marker, side=MARKER_SIDE):
    """Camera -> robot transform (OpenCV axes) from one detection.

    Returns (T, rms_px, rvec, tvec) where rvec/tvec is marker -> camera (for drawing).
    """
    obj = marker_corners(side)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, pixels, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise ValueError("Marker pose estimation failed")
    R, _ = cv2.Rodrigues(rvec)
    T_cam_marker = np.eye(4)
    T_cam_marker[:3, :3] = R
    T_cam_marker[:3, 3] = tvec.ravel()
    T = T_robot_marker @ np.linalg.inv(T_cam_marker)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - pixels) ** 2, axis=1))))
    if T[2, 3] <= 0:
        raise ValueError(
            f"Camera would be below the table (z = {T[2, 3]:.3f} m): wrong marker position / "
            "orientation (--marker-yaw), or marker not lying face up")
    return T, rms, rvec, tvec


def average_transforms(transforms):
    """Mean of several rigid transforms (mean translation, rotation re-projected by SVD)."""
    Ts = np.asarray(transforms)
    U, _, Vt = np.linalg.svd(Ts[:, :3, :3].mean(axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = Ts[:, :3, 3].mean(axis=0)
    return T


def capture(device, frames, width, height):
    cap = cv2.VideoCapture(int(device) if str(device).isdigit() else device)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise ValueError(f"Cannot open camera {device}")
    out = []
    try:
        for _ in range(10):  # let the auto-exposure settle
            cap.read()
        for _ in range(frames):
            ok, frame = cap.read()
            if ok:
                out.append(frame)
    finally:
        cap.release()
    if not out:
        raise ValueError("The camera returned no frame")
    return out


def load_intrinsics(args):
    if args.intrinsics:
        data = json.loads(Path(args.intrinsics).read_text())
        K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(data.get("D", []), dtype=np.float64)
        size = tuple(data["image_size"]) if "image_size" in data else None
        return K, D, size, data.get("intrinsic_rms_px")
    if args.chessboard_images:
        if not (args.cols and args.rows and args.square):
            raise ValueError("--chessboard-images needs --cols, --rows and --square")
        K, D, size, rms, _ = calibrate_intrinsics(args.chessboard_images, args.cols, args.rows, args.square)
        return K, D.ravel(), size, rms
    raise ValueError("Give --intrinsics (JSON) or --chessboard-images")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", nargs="+", help="picture(s) of the marker, taken by the real camera")
    src.add_argument("--device", help="camera index or /dev/videoN: captures --frames frames")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--intrinsics", help="JSON with K, D (and image_size), e.g. a calibrate_camera.py output")
    p.add_argument("--chessboard-images", nargs="+", help="alternative: chessboard pictures")
    p.add_argument("--cols", type=int, help="chessboard internal corner columns")
    p.add_argument("--rows", type=int, help="chessboard internal corner rows")
    p.add_argument("--square", type=float, help="chessboard square side in metres")
    p.add_argument("--marker-x", type=float, required=True,
                   help="x of the marker CENTRE in base_link, metres (forward)")
    p.add_argument("--marker-y", type=float, default=0.0, help="y of the marker centre (left), default 0")
    p.add_argument("--marker-z", type=float, default=0.0, help="z of the marker plane, default 0 (table)")
    p.add_argument("--marker-yaw", type=float, default=0.0,
                   help="direction of the marker's top edge, degrees CCW from +x (default 0: away from robot)")
    p.add_argument("--marker-id", type=int, default=MARKER_ID)
    p.add_argument("--marker-side", type=float, default=MARKER_SIDE, help="metres, default 0.100")
    p.add_argument("--frame-id", default="base_link")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    output = Path(args.output)
    if output.exists():
        p.error("Output already exists; choose another filename")
    try:
        K, D, size, intr_rms = load_intrinsics(args)
        if args.image:
            images = []
            for name in args.image:
                img = cv2.imread(name)
                if img is None:
                    raise ValueError(f"Cannot read {name}")
                images.append(img)
        else:
            images = capture(args.device, args.frames, args.width, args.height)
        shape = (images[0].shape[1], images[0].shape[0])
        if size is not None and shape != tuple(size):
            raise ValueError(f"Images are {shape[0]}x{shape[1]} but the intrinsics are for {size[0]}x{size[1]}: "
                             "use --width/--height or the matching intrinsics")

        T_rm = marker_to_robot(args.marker_x, args.marker_y, args.marker_yaw, args.marker_z)
        poses, rmss, last = [], [], None
        for img in images:
            corners = detect_marker(img, args.marker_id)
            if corners is None:
                continue
            T, rms, rvec, tvec = estimate_camera_pose(corners, K, D, T_rm, args.marker_side)
            poses.append(T)
            rmss.append(rms)
            last = (img, corners, rvec, tvec)
        if not poses:
            raise ValueError(f"Marker id {args.marker_id} (DICT_4X4_1000) not found in any of "
                             f"{len(images)} image(s): in view, lit, not blurred, not cut off?")
        T = average_transforms(poses)
        spread = float(np.std([t[:3, 3] for t in poses], axis=0).max() * 1000) if len(poses) > 1 else 0.0

        img, corners, rvec, tvec = last
        dbg = img.copy()
        cv2.polylines(dbg, [corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        cv2.drawFrameAxes(dbg, K, D, rvec, tvec, args.marker_side / 2)
        cv2.imwrite(str(output.with_suffix(".debug.png")), dbg)

        data = dict(K=K.tolist(), D=np.asarray(D).ravel().tolist(), T=T.tolist(),
                    image_size=list(shape), camera_axes="opencv", frame_id=args.frame_id,
                    intrinsic_rms_px=intr_rms, pose_rms_px=float(np.mean(rmss)),
                    marker=dict(dictionary="DICT_4X4_1000", id=args.marker_id, side_m=args.marker_side,
                                centre_robot=[args.marker_x, args.marker_y, args.marker_z],
                                yaw_deg=args.marker_yaw, frames_used=len(poses)))
        output.write_text(json.dumps(data, indent=2) + "\n")
    except (ValueError, KeyError, OSError, cv2.error) as exc:
        p.error(str(exc))
    c = T[:3, 3]
    print(f"Saved {output}: {len(poses)}/{len(images)} frame(s) used, reprojection RMS {np.mean(rmss):.2f} px, "
          f"frame-to-frame spread {spread:.1f} mm")
    print(f"Camera position in {args.frame_id}: x={c[0]:.3f} y={c[1]:.3f} z={c[2]:.3f} m")
    print(f"Check {output.with_suffix('.debug.png')}: the green square must sit on the marker, "
          "axes: red = marker x (right), green = y (top edge), blue = z (up).")


if __name__ == "__main__":
    main()
