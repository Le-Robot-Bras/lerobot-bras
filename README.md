# Tek5 — SO-ARM101 pick & place

Simulation, IK, perception and pick & place on the SO-ARM101 arm, with ROS2 and Docker.

- Project spec: [`docs/SUJET.md`](docs/SUJET.md)
- Setup and first steps: see the bootstrap slides.

## Phase 1: pick & place (sim)

One command (starts the stack if needed, new random cube, runs one pick & place, follows the state):

```bash
./docker/pick_place.sh          # RViz: http://localhost:6080/vnc.html?autoconnect=true&resize=scale
./docker/pick_place.sh --same          # keep the cube where it is
./docker/pick_place.sh --repeat 5      # 5 pick & place in a row (new cube each time) + summary
./docker/pick_place.sh --log           # live log (state, TCP, gripper, perception), saved in logs/
./docker/pick_place.sh --down          # stop the project (Docker Desktop keeps running)
./docker/pick_place.sh --stop          # stop the project and quit Docker Desktop
```

### Linux / macOS / Windows

`docker/pick_place.sh` detects the platform (`./docker/pick_place.sh --info` shows what it detected):

| | Docker | RViz |
|---|---|---|
| **Linux** | system engine (`systemctl`), never stopped by `--stop` | native X11 window: `rviz2` inside the `control` container, after `xhost +local:docker` (done by the script). Browser (noVNC) only with `--vnc` or without `$DISPLAY` (SSH) |
| **macOS** | Docker Desktop, started / quit by the script | browser (noVNC): Docker Desktop has no X11/GPU |
| **Windows** | Docker Desktop + WSL2 (run the script **from WSL**) | browser (noVNC) |

**Windows setup (once):**
1. Install WSL2 + Ubuntu: `wsl --install -d Ubuntu` in an admin PowerShell, then reboot.
2. Install Docker Desktop, enable *Settings > General > Use the WSL 2 based engine* and
   *Settings > Resources > WSL integration > Ubuntu*.
3. In the **Ubuntu** terminal (not PowerShell / Git Bash), clone the repo in the Linux filesystem
   (`~/`, not `/mnt/c/...`: much faster, and no CRLF line endings breaking the scripts):
   `git clone <repo> ~/lerobot-bras && cd ~/lerobot-bras && cp docker/.env.example docker/.env`
4. `./docker/pick_place.sh`, then open http://localhost:6080/vnc.html in the Windows browser.

Manually:

```bash
cd docker
docker compose up -d control perception viz        # driver + brain, perception, robot_state_publisher
docker compose exec control bash -lc "source /ros2_ws/install/setup.bash && \
    ros2 service call /brain/start std_srvs/srv/Trigger"
docker compose exec control bash -lc "source /ros2_ws/install/setup.bash && ros2 topic echo /brain_state"
```

| Node | Role |
|---|---|
| `so101_driver` | backend (sim / real / remote), `/joint_states` 30 Hz, `/joint_command` (clamped, partial OK), `/external_cam/image_raw` + `camera_info` 10 Hz |
| `so101_perception` | mask (HSV thresholds, or YOLO-seg with `--yolo`) + fit of a known-size box model → `/object_position`, `/drop_box_position` (position + yaw), `/perception/mask`, `/perception/debug`, static TF `base_link → external_cam` (the camera pose it uses: calibration file, else the driver's) |
| `so101_brain` | IK (pinocchio, `kinematics.py`) + state machine (`pick_place.py`), `/brain/start`, `/brain/stop`, `/brain_state`, `/go_to_target`, `/end_effector_pose` |

State machine: `IDLE → HOME → OBSERVE → APPROACH → DESCEND → GRASP → LIFT → TRANSPORT → RELEASE → RETREAT → DONE`,
with `RETRY` when a grasp fails (gripper not closed, or cube still on the table after the lift).

Evaluate without ROS (same code, MuJoCo, ground truth only used for scoring):

```bash
python3 tools/eval_pick_place.py --episodes 20        # success rate
python3 tools/eval_pick_place.py --ik-test 50         # IK error measured in MuJoCo (S4: < 1 cm)
python3 tools/eval_pick_place.py --perception-test 30 # perception error
python3 tools/eval_pick_place.py --episodes 1 --video run.mp4
python3 tools/eval_pick_place.py --perception-test 40 --detector yolo --lighting 0.2   # YOLO, dim light
```

### Perception with YOLO-seg (optional)

`./docker/pick_place.sh --yolo` (or `PERCEPTION_DETECTOR=yolo` in `docker/.env`) replaces the HSV
thresholds by a YOLO26n-seg network; the 3D model fit is unchanged. The network is trained on
synthetic images of the sim camera, labelled for free by MuJoCo's segmentation rendering, with
randomized lights, colours, arm poses and box position. Its coarse masks are refined at the
pixel by a colour model learnt in the current frame inside the YOLO mask (`vision.refine_mask`).

| Measured in sim | HSV | YOLO + refine |
|---|---|---|
| cube error, nominal light (40 scenes) | 0.8 mm / 2.2° | 0.9 mm / 2.5° |
| cube detected, 15 % light | 1 / 40 | 40 / 40 (0.8 mm) |
| pick & place, nominal light | 200 / 200 | 80 / 80 |
| pick & place, 20 % light | 0 / 40 | 40 / 40 |
| cost per image (cube + box) | ~10 ms | ~40 ms (Docker CPU: ~7 Hz) |

```bash
python3 tools/yolo_dataset.py --out datasets/so101_seg --images 6000   # images + exact labels
python3 tools/yolo_train.py --data datasets/so101_seg/data.yaml        # -> so101_perception/models/so101_seg.pt
python3 tools/yolo_live.py --run runs/yolo/so101_seg --data datasets/so101_seg   # live view, http://localhost:8008
```
Needs `ultralytics` (AGPL-3.0); the perception image installs it with a CPU-only PyTorch
(`--build-arg WITH_YOLO=0` to skip). Trained on sim images only: for the real camera, add real
labelled frames (or fine-tune) before trusting it.

### Demo

```bash
cd docker
docker compose -f docker-compose.yml -f docker-compose.vnc.yml up -d control perception viz rviz-vnc
# RViz in the browser: http://localhost:6080/vnc.html?autoconnect=true&resize=scale
docker compose exec control bash -lc "source /ros2_ws/install/setup.bash && ros2 run so101_brain demo_loop --runs 5"
```

`demo_loop` chains pick & places: `/sim/reset_object` (driver, sim only: new random cube) →
`/brain/start` → wait for `DONE`, and prints a summary. In RViz: the red cube is where the
**brain** believes it is (on the table, attached to the gripper, then in the box), the faint pink
cube is the raw perception, the spheres are the planned pre-grasp / grasp / drop points and the
text above the arm is the brain state.

Offline demo video (no ROS): `python3 tools/eval_pick_place.py --seeds 0,3,5 --demo-video demo.mp4`

## Phase 2: dance to the music, in real time (macOS / Linux)

Whatever plays on the Mac (Spotify, YouTube, a file...) is captured live and the arm dances to it:
beat tracking (predictive beat clock, so moves land on the beat), genre → dance style (electronic,
hip-hop, rock, pop, latin, calm), vocals and instruments (the gripper "sings"), drops / build-ups /
breaks. MuJoCo twin in the browser, optionally the real arm. No ROS, no Docker: everything runs on
the host. Code: `tools/dance_live.py` (entry point) → `tools/dancelive/` (audio, analysis, tempo,
genre, vocals, moves, choreo, safety, app).

### One-time setup (macOS)

1. **Loopback driver** (macOS cannot capture its own output without one):
   `brew install --cask blackhole-2ch`, then reboot.
2. **Output = your headphones/speakers + BlackHole**, created and selected as the default output:
   ```bash
   swift tools/macos_multi_output.swift                          # AirPods (default) + BlackHole
   swift tools/macos_multi_output.swift "External Headphones"    # wired headphones + BlackHole
   swift tools/macos_multi_output.swift "MacBook Pro Speakers"   # Mac speakers + BlackHole
   swift tools/macos_multi_output.swift --restore AirPods        # back to a plain output
   ```
   It creates a multi-output device "Dance Output". Volume keys don't act on a multi-output
   device: set the volume in the music app. Re-run it after reconnecting Bluetooth headphones.
3. **Microphone permission** for your terminal app (System Settings › Privacy & Security ›
   Microphone): BlackHole counts as a microphone; without it the capture is pure silence. Restart
   the terminal app after granting it.
4. Check: `uv run --no-project --python 3.12 --with sounddevice --with numpy python tools/dance_live.py --list-devices`
   must list `BlackHole 2ch`.

### One-time setup (Linux: Ubuntu / Debian, PipeWire or PulseAudio)

No loopback driver is needed: PipeWire and PulseAudio expose a **monitor** source for every
output (a copy of what you hear).

1. System packages (PortAudio for `sounddevice`, ffmpeg to decode test files, the ALSA → Pulse
   bridge so PortAudio sees the `pulse` device):
   ```bash
   sudo apt install libportaudio2 ffmpeg pulseaudio-utils libasound2-plugins
   # PipeWire systems (Ubuntu >= 22.10): also `sudo apt install pipewire-pulse`
   curl -LsSf https://astral.sh/uv/install.sh | sh     # uv, if not installed
   ```
2. Point the capture at the monitor of the current output and check that `pulse` is listed:
   ```bash
   pactl get-default-sink                                   # e.g. alsa_output.pci-0000_00_1f.3.analog-stereo
   export PULSE_SOURCE="$(pactl get-default-sink).monitor"  # put it in ~/.bashrc to keep it
   uv run --no-project --python 3.12 --with sounddevice --with numpy python tools/dance_live.py --list-devices
   ```
   Then use `--device pulse` instead of `--device "BlackHole 2ch"` in every command below.
   If you change the output (headphones plugged in, Bluetooth), re-run the `export` line.
   Check the capture works: play music and look at `loud` in the status line (about −30…−10 dB,
   `-120 dB` = silence: wrong source).
3. Bluetooth headphones: the monitor is captured *before* the Bluetooth delay, so add
   `--headphones airpods` (or `--output-delay 0.2`) so the arm matches what you hear.
4. Headless machine / SSH: the twin view renders off-screen; if MuJoCo can't open a GL context,
   prefix the commands with `MUJOCO_GL=egl` (NVIDIA/Intel) or `MUJOCO_GL=osmesa` (CPU, `sudo apt
   install libosmesa6`). The view is still served on http://localhost:8011.
5. Real arm: the serial port is `/dev/ttyACM0` (find it with `ls /dev/ttyACM*`); add yourself to the
   `dialout` group once (`sudo usermod -aG dialout $USER`, then log out/in). The arm server runs on
   the host exactly as on macOS (`--port /dev/ttyACM0`).
6. Demucs (vocals) uses an NVIDIA GPU if PyTorch sees one, else the CPU (~1 s per 6-s chunk: fine
   live). Without a GPU you can also pass `--no-vocal-model` to save CPU.

Python 3.12 is required (librosa/numba, demucs). All commands below use:
```bash
DANCE="uv run --no-project --python 3.12 --with mediapipe --with demucs --with torchaudio \
  --with sounddevice --with numpy --with mujoco --with opencv-python-headless python tools/dance_live.py"
```
The first run downloads YAMNet (genre, ~4 MB) and Demucs (vocals, ~80 MB) to `~/.cache`.

### Simulation (no arm)

```bash
$DANCE --device "BlackHole 2ch"      # macOS: then play music; view: http://localhost:8011
$DANCE --device pulse                # Linux (with PULSE_SOURCE=<sink>.monitor, see above)
$DANCE --file song.mp3               # test with a file (plays it and dances)
$DANCE --file song.mp3 --fast --no-view --log out.npz --verbose   # headless, faster than real time
```
The view shows the SIM arm next to the TARGET, the style and music BPM, beat pulse, section,
current move, vocals (`VOX sung 0.8`) and instruments (`INST DRUMS BASS GTR`).

### Real arm

1. Arm server on the host, motion allowed. `--max-step 0.15` is needed (with the default 0.05 the
   elbow is too weak to lift the forearm); `--servo-gain 32` makes the servos stiffer and snappier
   (LeRobot's default 16 is soft; lower it if the arm shakes):
   ```bash
   # --port: macOS /dev/cu.usbmodemXXXX, Linux /dev/ttyACM0
   uv run --with ./core/lerobot_min tools/real_arm_server.py --port /dev/cu.usbmodemXXXX \
     --calibration ~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json \
     --allow-motion --max-step 0.15 --servo-gain 32 --servo-accel 254
   ```
2. Clear the space around the arm, keep a hand near it, then:
   ```bash
   $DANCE --device "BlackHole 2ch" --real                         # safe limits: 1.5 rad/s, 12 rad/s²
   $DANCE --device "BlackHole 2ch" --real --vmax 3.0 --amax 60    # livelier (tested)
   ```
   (Linux: `--device pulse` instead of `--device "BlackHole 2ch"`.)
   The arm stiffens, goes slowly to its dance pose and waits for music; when the music stops it
   returns to the base pose. Ctrl-C (or a joint lagging its target by more than `--max-lag`)
   holds the present pose. When the dance program exits, the server keeps the arm stiff where it
   is; **hold the arm before stopping the server** (it goes limp).
3. Check the calibration first if the arm was reassembled or recalibrated elsewhere: run the server
   without `--allow-motion` and compare the twin with the real arm by moving it by hand. A motor
   mounted reversed takes `"drive_mode": 1` in the calibration JSON, and a joint whose zero is not
   the middle of its range takes `"zero_offset": <rad>` (both supported in RADIANS mode by
   `core/lerobot_min`).

Real-arm defaults that differ from the sim: `--vmax 1.5`, `--amax 12`, gripper still unless
`--gripper` (its calibration is fragile), `--servo-lead 0.08` s (servo response compensated on the
beat clock).

### Useful options

| Option | Effect |
|---|---|
| `--style rock` | force a style instead of detecting the genre |
| `--amp 0.6` | smaller moves |
| `--headphones airpods` | Bluetooth delay (~180 ms) compensated so the arm matches what you hear; trim with `--output-delay 0.2` |
| `--latency 0.07` | beat clock further ahead if the arm looks late |
| `--no-genre` / `--no-vocal-model` | skip the genre / Demucs models (lighter) |
| `--check-moves` | check every move against joint limits and collisions, then exit |

`tools/latency_calibrate.py` measures the output → BlackHole delay with clicks (stop the dance
first: it plays sound and reads BlackHole).

### Vocal / instrument model

A small student model (`~/.cache/so101/vocal_model.npz`) detects vocals, drums, bass and melodic
instruments live; it is distilled from Demucs separations of songs in a folder. Retrain after
adding songs (only new songs are processed):
```bash
uv run --no-project --python 3.12 --with demucs --with torchaudio --with mediapipe --with numpy \
  python tools/train_vocal_model.py --songs ~/Desktop/Anime/music
```

### Offline dance to a file (pre-checked, no live analysis)

`tools/dance.py --sim|--real --music song.mp3` analyses the whole song first (librosa beats and
loudness), builds the choreography and checks it entirely (limits, speed, collisions) before
anything moves; `--video out.mp4` renders it with the song. From Python:
`from dance import dance_to_music; dance_to_music("song.mp3", mode="real", amp=0.6)`.
