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
| `so101_perception` | HSV mask + fit of a known-size box model → `/object_position`, `/drop_box_position` (position + yaw), `/perception/mask`, `/perception/debug` |
| `so101_brain` | IK (pinocchio, `kinematics.py`) + state machine (`pick_place.py`), `/brain/start`, `/brain/stop`, `/brain_state`, `/go_to_target`, `/end_effector_pose` |

State machine: `IDLE → HOME → OBSERVE → APPROACH → DESCEND → GRASP → LIFT → TRANSPORT → RELEASE → RETREAT → DONE`,
with `RETRY` when a grasp fails (gripper not closed, or cube still on the table after the lift).

Evaluate without ROS (same code, MuJoCo, ground truth only used for scoring):

```bash
python3 tools/eval_pick_place.py --episodes 20        # success rate
python3 tools/eval_pick_place.py --ik-test 50         # IK error measured in MuJoCo (S4: < 1 cm)
python3 tools/eval_pick_place.py --perception-test 30 # perception error
python3 tools/eval_pick_place.py --episodes 1 --video run.mp4
```

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
