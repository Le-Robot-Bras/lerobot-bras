# Real arm + phone camera on macOS: setup, from chessboard to pick & place

Everything to run the pick & place on the real SO-ARM101 with a phone as the camera, from a Mac
(Docker Desktop). Run commands in a terminal (zsh) from the repo root unless a section says
otherwise. For Linux/Windows see the platform table in `README.md`.

Sim figures in the README do not carry over automatically: the real pipeline (camera, HSV
thresholds, grasp) has only been tuned in simulation. Expect to tune.

## How it fits together on a Mac

Docker Desktop cannot hand a USB serial port to a container, so the arm is served from the Mac:

```
Mac terminal A:  tools/real_arm_server.py  --USB-->  arm
Container:       driver (REMOTE_ARM)  --TCP host.docker.internal:5555-->  server
Phone:           http://<phone-ip>:8080/video  --Wi-Fi-->  driver container (camera) / Mac (calibration)
```

You therefore keep **two** things running: the arm server (terminal A, stays open) and the
Docker stack (`docker compose`, terminal B).

## 0. Prerequisites (once)

- **Docker Desktop** installed and running (`open -a Docker`; wait until the whale icon says running).
- **uv**: `brew install uv` (or `curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **Arm connected by USB**. Find its port (use `cu.`, not `tty.`):
  ```bash
  ls /dev/cu.usbmodem*
  ```
  The name can change when you re-plug the arm.
- **Arm calibrated** (once per arm): see step 1.
- **Phone camera app** serving a stream (e.g. IP Webcam: `http://<phone-ip>:8080/video`). Phone and
  Mac on the same Wi-Fi. Video resolution **1280x720**, stabilisation and digital zoom **off**,
  focus locked if possible.
- **macOS Local Network permission**: if the Mac cannot reach the phone ("No route to host"),
  allow Terminal and Docker under *System Settings > Privacy & Security > Local Network*.

## 1. Calibrate the arm (once per arm)

Needed once: `real_arm_server.py` and the driver refuse an arm whose motors do not match the
calibration JSON. **If a teammate already calibrated this arm, copy their JSON to
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/` and skip this step**:
re-calibrating overwrites the values stored in the motors and breaks their file.

The script is interactive: you move the arm **by hand**, so run it yourself in a terminal.

1. Stop anything that holds the serial port (the arm server, the Docker stack):
   ```bash
   docker compose -f docker/docker-compose.yml -f docker/docker-compose.vnc.yml stop
   ```
2. Find the port and run the calibration:
   ```bash
   ls /dev/cu.usbmodem*
   uv run --with ./core/lerobot_min tools/calibrate_arm.py --port /dev/cu.usbmodemXXXX
   ```
   Options: `--id <name>` (file name, default `my_awesome_follower_arm`), `--force` to overwrite an
   existing file, `--yes` to skip the confirmation.
3. Type `yes` at the warning. Torque is switched **off**: the arm goes limp, so **support it** and
   keep its path clear.
4. **Middle pose**: put every joint in the middle of its range (upper arm vertical, forearm
   horizontal; **not** the folded rest pose) and press ENTER. This writes the homing offsets into
   the motors.
5. **Ranges**: move every joint **except `wrist_roll`** slowly through its **full range** (both
   ends), then press ENTER to stop recording.
6. The script saves the JSON and prints the recorded range of each joint. Each should be close to
   the expected span (`shoulder_pan` ~220 deg, `shoulder_lift` ~200, `elbow_flex` ~194,
   `wrist_flex` ~190). A line marked `SUSPICIOUS` means that joint was not moved through its full
   range, or the middle pose was set at an extreme: redo it with `--force`. It must also print
   `Motors match the file: YES`.

Result: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json`.
This is the file `CALIBRATION_FILE` points to in step 2 and `--calibration` in step 6.

If `CALIBRATION_FILE` in `docker/.env` already points to a missing path, Docker may have created an
empty *directory* there: remove it with `rmdir <path>` before calibrating.

## 2. Configure `docker/.env`

```bash
cp docker/.env.example docker/.env     # first time only
```

Set (use **absolute paths**, e.g. `/Users/<you>/...`):

```
USE_SIM=false
REMOTE_ARM=host.docker.internal:5555              # arm served by real_arm_server.py on the Mac
ARM_TORQUE=true                                   # the arm follows commands: clear the workspace first
CALIBRATION_FILE=/Users/<you>/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json
CAMERA_DEVICE=http://<phone-ip>:8080/video        # http, not https, unless you enabled TLS in the app
CAMERA_CALIBRATION=/Users/<you>/calib/camera.json # filled in at step 5
PERCEPTION_DETECTOR=hsv
BRAIN_AUTO_START=false
```

`REAL_ROBOT_PORT` is not used in this mode (the server holds the serial port, not the container).
Keep `CALIBRATION_FILE` pointing at an existing file: if it points to a missing path, Docker
creates an empty *directory* there (`rmdir` it). The file must be under a folder shared with
Docker Desktop (`/Users` is by default).

`.env` is only read when containers are created: after any edit run
`docker compose up -d --force-recreate control perception viz` (from `docker/`).

If the phone's IP changes, edit `CAMERA_DEVICE` and recreate. The calibration stays valid
(it depends on the phone's position and resolution, not its IP).

## 3. Test the stream

```bash
uv run --with opencv-python-headless python3 -c "
import cv2
c = cv2.VideoCapture('http://<phone-ip>:8080/video')
ok, f = c.read(); print(ok, None if f is None else f.shape)"
```

Expect `True (720, 1280, 3)`.

you can also open the adress in browser without `/video` to modify camera settings

don't use tech's wify or it will fuck up the lan

## 4. Camera intrinsics: chessboard

Print a chessboard (e.g. calib.io, 10x7 squares of 25 mm = **9x6 internal corners**), print at 100 %
scale, check a square with a ruler, glue it flat on a rigid board.

Fix the phone where it will stay for the whole session, then grab 20 frames from the stream
(not from the phone's photo mode: the resolution must match the stream):

```bash
uv run --with opencv-python-headless tools/grab_frames.py \
  --device http://<phone-ip>:8080/video --out ~/calib/chess --count 20 --interval 3
```

Move the board between shots: tilt up to ~45 deg, reach the corners of the image, vary the distance,
hold still when a frame is taken. At least 10 frames must contain a detected board.

## 5. Camera pose: ArUco marker

Marker: `DICT_4X4_1000`, **id 0**, **100 mm** side (outer edge of the black border, i.e. the whole
printed square).

1. Lay the marker **flat, face up**, with a clear white margin all around it (nothing touching it).
   Its **top edge points away from the robot** (reads upright when you stand behind the robot).
2. Measure the **centre** of the marker in the robot frame `base_link`: `x` forward, `y` to the left,
   in metres, from the base origin. This number goes straight into every later position, so measure
   it carefully.
3. Do not move the phone after this point.

```bash
uv run --with opencv-python-headless --with numpy \
  python3 ros2_ws/src/so101_perception/so101_perception/calibrate_camera_aruco.py \
  --chessboard-images ~/calib/chess/*.png --cols 9 --rows 6 --square 0.025 \
  --device http://<phone-ip>:8080/video --frames 20 \
  --marker-x 0.23 --marker-y 0 \
  --output ~/calib/camera.json
```

(`0.23` is the value measured for our setup: replace it with yours.)

It prints the intrinsic error, the reprojection error and the camera position in `base_link`.
Checks:

- Intrinsic RMS should be below about 1 px; pose RMS about 1 px.
- **Open `~/calib/camera.debug.png`** (`open ~/calib/camera.debug.png`): the green square must sit on
  the marker; blue axis up, green axis along the marker's top edge (away from the robot).
- The printed camera position must match reality with a ruler.
- Optional: project the robot base into the image. If the base frame does not land on the real base,
  the marker position/orientation you gave is wrong.

To redo only the pose (phone untouched, marker moved or re-measured), reuse the intrinsics already
stored in `camera.json` and write a **new** output file (existing files are never overwritten):

```bash
uv run --with opencv-python-headless --with numpy \
  python3 ros2_ws/src/so101_perception/so101_perception/calibrate_camera_aruco.py \
  --intrinsics ~/calib/camera.json --device http://<phone-ip>:8080/video --frames 20 \
  --marker-x 0.23 --marker-y 0 --output ~/calib/camera2.json
```

Then **remove the marker** from the table (it lies inside the cube's working area). **If the phone
moves, redo this step.** Keep a copy of `camera.json`.

## 6. Start the arm server (terminal A, leave it open)

First a read-only check (nothing can move; stop it with Ctrl-C once it connects):

```bash
uv run --with ./core/lerobot_min tools/real_arm_server.py \
  --port /dev/cu.usbmodemXXXX \
  --calibration ~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json
```

Expect `arm connected on ... (READ-ONLY)`. Then start it for real with motion allowed:

```bash
uv run --with ./core/lerobot_min tools/real_arm_server.py \
  --port /dev/cu.usbmodemXXXX \
  --calibration ~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json \
  --allow-motion
```

Expect `arm connected on ... (motion ALLOWED)`. Safety built into the server:

- goals are clipped to `--max-step` (0.05 rad) of the present position per command;
- enabling torque first holds the present position (no jump);
- if the driver goes silent for `--watchdog` seconds (1 s) or disconnects, the arm freezes where it is;
- torque is released when the server exits: **the arm goes limp, hold it** before stopping the server.

Only one program may hold the serial port: no other terminal, calibration script or server instance.

## 7. Start the stack (terminal B)

```bash
cd docker
docker compose up -d --force-recreate control perception viz
sleep 25
docker compose logs control perception --tail 60 | grep -i -E "camera|connected|torque|calibration|error|Traceback"
```

Expect:

- `Robot connected: SO101Remote` and `Remote arm: torque ENABLED` (the arm becomes firm: keep your hand near it).
- `Camera 'http://...' opened (1280x720)`.
- `Camera calibration loaded from /calibration/camera.json`.

The first start builds the images and the ROS workspace (several minutes).

## 8. Open a shell and check the nodes

```bash
docker compose exec control bash
```

Inside the container (every new shell):

```bash
source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash
ros2 node list
timeout 5 ros2 topic echo /joint_states --once
timeout 5 ros2 topic echo /end_effector_pose --once
```

Always put `timeout 5` before `ros2 topic echo --once`: it hangs forever if nothing is published.
(`timeout` exists inside the container; it does not exist in the Mac's own shell.)

## 9. Prepare the scene

- **Cube**: 5 cm, **red** (the HSV thresholds are tuned for red). On the table, within reach of the
  arm (roughly 12-32 cm from the base) and fully visible by the camera.
- **Drop box**: brown, about 10.4 cm outer width, 5 cm walls, on the table, reachable and visible.
- Even lighting, no strong shadows over the cube. A brick or any other shape is not supported: the
  perception and the grasp are built for the 5 cm cube.

## 10. Check the perception against a tape measure

In the container:

```bash
timeout 5 ros2 topic echo /object_position --once
timeout 5 ros2 topic echo /drop_box_position --once
```

Measure the real centres (x forward, y left, from the base origin) and compare. Under about 1 cm
is good. If nothing is published, the mask does not see the object (colour/lighting); if the error
is large, re-check the calibration (step 5).

## 11. Dry run: move above the cube without touching it

Using the cube position from step 10 (`X`, `Y`), keep the gripper well above it:

```bash
ros2 service call /go_to_target so101_interfaces/srv/GoToTarget \
  "{target_pose: {header: {frame_id: 'base_link'}, pose: {position: {x: X, y: Y, z: 0.12}}}}"
```

The gripper must end up directly above the cube. If it is offset, fix the calibration before going
further. Unreachable targets are refused with a message.

## 12. Pick & place

Open a third terminal (C) to follow the state:

```bash
cd docker
docker compose exec control bash
source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash
ros2 topic echo /brain_state
```

In the container shell of step 8, with the workspace clear and a hand ready:

```bash
ros2 service call /brain/start std_srvs/srv/Trigger
```

State machine: `HOME -> OBSERVE -> APPROACH -> DESCEND -> GRASP -> LIFT -> TRANSPORT -> RELEASE -> RETREAT -> DONE`
(`RETRY` if the grasp fails).

**Abort at any time:**

```bash
ros2 service call /brain/stop std_srvs/srv/Trigger
```

Do not use `./docker/pick_place.sh` for the real arm: it resets the simulated cube.

## 13. Shut down

1. Hold the arm (it goes limp when the server exits).
2. Terminal B: `docker compose down` (from `docker/`).
3. Terminal A: Ctrl-C to stop the arm server.
4. Optional: quit Docker Desktop.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no configuration file provided: not found` | Run from `docker/`, or `docker compose -f docker/docker-compose.yml ...`. |
| `Cannot connect to the Docker daemon` | Docker Desktop not running: `open -a Docker` and wait. |
| `No such file or directory` for `/dev/cu.usbmodem...` | Arm unplugged or port renamed: `ls /dev/cu.usbmodem*`. |
| `Resource busy` / port in use | Another process holds the serial port (a second server, the calibration script, the stack on another setup). Stop it. |
| Driver: connection refused to `host.docker.internal:5555` | Arm server (terminal A) not running, wrong `--listen` port, or `REMOTE_ARM` mistyped. Start the server first, then recreate `control`. |
| Arm limp, joints read but not moving | Server started without `--allow-motion`, or `ARM_TORQUE` not `true` (then recreate `control`). |
| `does not match the motors` | Wrong arm calibration JSON for this arm: recalibrate (`--force`) or copy the right file. |
| `No calibration file at /calibration/arm.json` / `CALIBRATION_FILE` is a directory | Path wrong or file missing: use the absolute path; if Docker created an empty directory there, `rmdir` it. |
| `ros2: command not found` | Source both setup files (step 8). |
| `ros2 topic echo` hangs | Nothing is published: check `docker compose logs control`. |
| `Cannot open camera` | Wrong/changed `CAMERA_DEVICE` (http vs https, IP), Local Network permission denied, or `.env` edited without recreating the containers. |
| Chessboard not detected in many frames | Dim or low-contrast print, strong tilt: the script falls back to the classic OpenCV detector. Reshoot with more light/contrast if still < 10. |
| `Marker id 0 ... not found` | Marker partly covered, white margin hidden, too small/oblique, motion blur, or wrong dictionary/id. |
| `Camera would be below the table` | Marker position/orientation or `--marker-yaw` wrong (or the marker is not face up). |
| Cube not detected | Not red, poor light, partly out of view. Brick/other shapes are not supported. |
| Gripper offset above the cube in step 11 | Calibration error: re-measure `--marker-x/--marker-y`, redo step 5; phone may have moved. |
