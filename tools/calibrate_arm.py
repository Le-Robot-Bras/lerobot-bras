#!/usr/bin/env python3
"""calibrate_arm.py - calibrates the real SO-ARM101 and saves the calibration file.

Interactive: you move the arm BY HAND while the script guides you. Run it yourself
in a terminal (it waits for ENTER); it cannot run unattended.

    uv run --with ./core/lerobot_min tools/calibrate_arm.py --port /dev/cu.usbmodemXXXX

What it does (SO101Follower.calibrate(), the procedure shipped with the project):
  1. disables torque on the 6 motors (the arm goes limp: SUPPORT IT),
  2. you put the arm in the middle of its range -> writes the homing offsets INTO THE MOTORS,
  3. you move every joint (except wrist_roll) through its full range -> records min/max,
  4. writes the ranges into the motors and saves the JSON file.

It overwrites the calibration stored in the motors. If a teammate already calibrated
this arm, copy their JSON file instead: re-calibrating makes theirs stop matching.
It never enables torque and never commands a motion.
"""
import argparse
from pathlib import Path

DEFAULT_DIR = "~/.cache/huggingface/lerobot/calibration/robots/so_follower"
DEFAULT_ID = "my_awesome_follower_arm"  # matches docker/.env.example (CALIBRATION_FILE)

# Full travel of each joint in degrees, from the URDF limits (so101.urdf).
EXPECTED_SPAN_DEG = {
    "shoulder_pan": 220.0,
    "shoulder_lift": 200.0,
    "elbow_flex": 193.6,
    "wrist_flex": 190.0,
}  # wrist_roll is a full turn by design (range forced to 0-4095): not checked


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="serial port of the arm, e.g. /dev/cu.usbmodemXXXX")
    p.add_argument("--id", default=DEFAULT_ID, help="calibration name, file is <id>.json (default %(default)s)")
    p.add_argument("--calibration-dir", default=DEFAULT_DIR, help="where to save (default %(default)s)")
    p.add_argument("--force", action="store_true", help="overwrite an existing calibration file")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = p.parse_args()

    calib_dir = Path(args.calibration_dir).expanduser()
    target = calib_dir / f"{args.id}.json"
    if target.is_dir():
        raise SystemExit(
            f"{target} is a DIRECTORY (Docker creates one when it mounts a missing file).\n"
            f"Remove it first:  rmdir '{target}'"
        )
    if target.is_file() and not args.force:
        raise SystemExit(f"{target} already exists. Use --force to overwrite it (and the motors' calibration).")

    # Imported here so --help works without the dependencies.
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(
        SO101FollowerConfig(port=args.port, id=args.id, calibration_dir=calib_dir, use_radians=True)
    )
    robot.calibration = {}  # force a fresh calibration even if --force replaces a file

    print(f"\nArm on {args.port}. The calibration will be saved to:\n  {target}\n")
    print("WARNING: this WRITES homing offsets and limits into the motors, and torque will be")
    print("DISABLED: the arm goes limp. Support it and make sure nothing is in its way.")
    if not args.yes and input("Type 'yes' to continue: ").strip().lower() != "yes":
        raise SystemExit("Aborted, nothing was written.")

    # bus.connect() only pings and reads. SO101Follower.connect() would call configure(),
    # which re-enables torque: not wanted here.
    robot.bus.connect()
    try:
        robot.calibrate()
        ok = robot.bus.is_calibrated
    finally:
        robot.bus.disconnect()  # disables torque

    if not target.is_file():
        raise SystemExit("Calibration finished but no file was written. Something went wrong.")
    print(f"\nSaved: {target}")
    print("Motors match the file:", "YES" if ok else "NO (re-run with --force)")

    # Sanity check: recorded span of each joint vs the URDF limits (joint zero = middle of the span).
    print("\nRecorded range per joint (should be close to the URDF span):")
    bad = False
    for name, cal in robot.calibration.items():
        span = (cal.range_max - cal.range_min) * 360 / 4095
        expected = EXPECTED_SPAN_DEG.get(name)
        if expected is None:
            continue
        # A full turn on one of these joints means the encoder wrapped around.
        suspicious = abs(span - expected) > 30 or span > 355
        bad |= suspicious
        print(f"  {name:14s} {span:6.0f} deg   (URDF: {expected:.0f})   {'<-- SUSPICIOUS' if suspicious else 'ok'}")
    if bad:
        print("\nA suspicious range usually means the 'middle' pose was set with that joint at an")
        print("extreme (e.g. the folded rest pose), or that it was not moved through its full range.")
        print("Redo it with --force: middle pose = upper arm vertical, forearm horizontal.")


if __name__ == "__main__":
    main()
