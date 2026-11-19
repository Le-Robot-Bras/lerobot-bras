#!/usr/bin/env python3
"""grab_frames.py - save frames from a camera / phone stream (same resolution as the driver sees).

    uv run --with opencv-python-headless tools/grab_frames.py --device http://IP:8080/video \
        --out chess --count 20 --interval 3

Move the chessboard between shots (tilt it, move it to the corners). Press nothing: it waits
--interval seconds before each frame. Used to take the chessboard images for the intrinsics.
"""
import argparse
import time
from pathlib import Path

import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", required=True, help="index, /dev/videoN or stream URL")
    p.add_argument("--out", required=True, help="output directory (created)")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--interval", type=float, default=3.0, help="seconds between frames")
    args = p.parse_args()

    cap = cv2.VideoCapture(int(args.device) if args.device.isdigit() else args.device)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.device}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    try:
        while saved < args.count:
            deadline = time.time() + args.interval
            while time.time() < deadline:  # keep reading so the stream buffer stays fresh
                cap.read()
            ok, frame = cap.read()
            if not ok:
                print("read failed, retrying")
                continue
            path = out / f"frame_{saved:02d}.png"
            cv2.imwrite(str(path), frame)
            saved += 1
            print(f"{saved}/{args.count} {path} ({frame.shape[1]}x{frame.shape[0]})")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
