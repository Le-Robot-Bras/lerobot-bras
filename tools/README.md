# tools

## real_arm_server.py - real arm on macOS (or any host without USB passthrough)

Docker Desktop on macOS cannot hand a USB serial port to a container. The server
runs on the host and owns the arm; the `control` container reaches it over TCP.

```
Mac:        real_arm_server.py  --USB-->  arm
Container:  driver_node (SO101Remote)  --TCP host.docker.internal:5555-->  server
```

The ROS stack does not notice: `SO101Remote` has the same API as `SO101Sim` and
`SO101Follower`.

### 1. Read-only check (nothing can move)

```bash
uv run --with ./core/lerobot_min tools/real_arm_server.py \
    --port /dev/cu.usbmodemXXXX \
    --calibration ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json
```

Find the port with `ls /dev/cu.usbmodem*`.

### 2. Point the driver at it

In `docker/.env`:

```
USE_SIM=false
REMOTE_ARM=host.docker.internal:5555
```

then `docker compose up -d control`. `/joint_states` now comes from the real arm.

### 3. Moving the arm (only when the area is clear)

Restart the server with `--allow-motion`, then set `ARM_TORQUE=true` in `docker/.env`
and restart `control`. Safety rules:

- goals are clipped to `--max-step` (0.05 rad) of the present position per command;
- enabling torque first holds the present position (no jump);
- if the driver goes silent for `--watchdog` seconds (1 s) or disconnects, the arm
  freezes where it is;
- torque is released when the server exits (the arm relaxes: hold it or park it
  in its rest pose first).

## calibrate_arm.py - create the arm's calibration file (interactive)

Needed once per arm: `real_arm_server.py` and the driver refuse an arm whose motors
do not match the calibration JSON. You move the arm BY HAND, so run it yourself:

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.vnc.yml stop   # nothing must hold the port
uv run --with ./core/lerobot_min tools/calibrate_arm.py --port /dev/cu.usbmodemXXXX
```

It writes homing offsets into the motors and saves
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`. If a teammate already
calibrated this arm, copy their JSON instead (re-calibrating breaks theirs).

Pitfall: if `CALIBRATION_FILE` in `docker/.env` points to a missing file, `docker compose up`
makes Docker create an empty DIRECTORY there. Remove it (`rmdir <path>`) before calibrating.

### Tests (no hardware)

```bash
python3 -m unittest discover -s tools
```
