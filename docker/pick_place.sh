#!/usr/bin/env bash
# pick_place.sh - one command for the SO-ARM101 pick & place in simulation.
#
#   ./docker/pick_place.sh                     # start Docker if needed, start the project cleanly,
#                                              # new random cube, one pick & place, follow the state
#   ./docker/pick_place.sh --same              # keep the cube where it is
#   ./docker/pick_place.sh --repeat 5          # 5 pick & place in a row (new cube each time) + summary
#   ./docker/pick_place.sh --log               # live log: brain state, TCP, gripper, perception
#                                              # (also saved in logs/pick_place_<date>.log)
#   ./docker/pick_place.sh --vnc               # force RViz in the browser (noVNC) instead of X11
#   ./docker/pick_place.sh --down              # stop the project (Docker keeps running)
#   ./docker/pick_place.sh --stop              # stop the project, and quit Docker Desktop (macOS/Windows)
#   ./docker/pick_place.sh --info              # show the detected platform / RViz mode, start nothing
#   Options can be combined: ./docker/pick_place.sh --repeat 3 --log
#
# Platforms:
#   Linux    RViz in a native X11 window (rviz2 inside the control container, as designed).
#            noVNC only with --vnc, or automatically when there is no $DISPLAY (SSH/headless).
#   macOS    Docker Desktop has no X11/GPU: RViz in the browser through noVNC.
#   Windows  run this script from WSL2 (Ubuntu) with Docker Desktop + WSL integration:
#            RViz in the browser through noVNC (works on every Windows/WSL version).
set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")" || exit 1

COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.vnc.yml)
ROS='source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash'
RVIZ_URL="http://localhost:6080/vnc.html?autoconnect=true&resize=scale"
UP_LOG="${TMPDIR:-/tmp}/pick_place_up.log"
MODE=""; REPEAT=1; LOG=0; SAME=0; FORCE_VNC=0
usage() { sed -n '2,21p' "$SCRIPT"; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --stop|--down|--info) MODE="$1" ;;
    --same) SAME=1 ;;
    --log) LOG=1 ;;
    --vnc) FORCE_VNC=1 ;;
    --repeat) shift; [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -ge 1 ] || usage; REPEAT="$1" ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1"; usage ;;
  esac
  shift
done
LOGFILE=""

log() { echo "[$(date +%T)] $*"; [ -n "$LOGFILE" ] && echo "[$(date +%T)] $*" >>"$LOGFILE"; return 0; }
die() { echo "[$(date +%T)] ERROR: $*" >&2; [ -n "$LOGFILE" ] && echo "[$(date +%T)] ERROR: $*" >>"$LOGFILE"; exit 1; }

# ------------------------------------------------------------------------ platform

case "$(uname -s)" in
  Darwin) PLATFORM=macos ;;
  Linux)  if grep -qi microsoft /proc/version 2>/dev/null; then PLATFORM=wsl; else PLATFORM=linux; fi ;;
  MINGW*|MSYS*|CYGWIN*) die "on Windows, run this script from WSL2 (Ubuntu), not Git Bash/PowerShell. See README." ;;
  *) PLATFORM=linux ;;
esac

# How RViz is shown: native X11 window on Linux with a display, noVNC (browser) otherwise.
if [ "$PLATFORM" = linux ] && [ "$FORCE_VNC" -eq 0 ] && [ -n "${DISPLAY:-}" ]; then
  VIEWER=x11
else
  VIEWER=vnc
fi
if [ "$VIEWER" = vnc ]; then
  SERVICES=(control perception viz rviz-vnc)
else
  SERVICES=(control perception viz)
fi

if [ "$MODE" = --info ]; then
  echo "platform: $PLATFORM | RViz: $VIEWER | services: ${SERVICES[*]} | DISPLAY=${DISPLAY:-<none>}"
  exit 0
fi

# ------------------------------------------------------------------------- Docker

engine_ok() { docker info >/dev/null 2>&1; }

wait_engine() {  # $1 = timeout (s)
  local t=0
  until engine_ok; do
    [ "$t" -ge "$1" ] && return 1
    sleep 2; t=$((t + 2))
  done
}

DESKTOP_PROC_MAC="Docker Desktop.app/Contents/MacOS/Docker Desktop"
WIN_DOCKER_EXE="/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe"

launch_desktop() {  # start Docker Desktop (macOS / Windows from WSL)
  docker desktop start --detach >/dev/null 2>&1 && return 0
  case "$PLATFORM" in
    macos) open -a Docker ;;
    wsl)   [ -x "$WIN_DOCKER_EXE" ] && ("$WIN_DOCKER_EXE" >/dev/null 2>&1 &) ;;
  esac
}

start_docker() {
  engine_ok && return 0
  case "$PLATFORM" in
    linux)
      # Native engine (systemd service). Docker Desktop for Linux is also handled.
      log "Docker engine not running: starting it (sudo may ask for your password)..."
      if systemctl --user list-unit-files docker-desktop.service >/dev/null 2>&1 \
          && systemctl --user start docker-desktop 2>/dev/null; then :;
      else
        sudo systemctl start docker || die "cannot start Docker: 'sudo systemctl start docker'"
      fi
      wait_engine 60 || die "Docker engine did not start (also check: groups | grep docker)."
      ;;
    macos|wsl)
      log "Starting Docker Desktop..."
      launch_desktop
      if ! wait_engine 120; then
        log "Docker Desktop does not answer, restarting it..."
        if ! docker desktop restart --detach >/dev/null 2>&1; then
          [ "$PLATFORM" = macos ] && { pkill -f "$DESKTOP_PROC_MAC" 2>/dev/null; sleep 5; }
          launch_desktop
        fi
        wait_engine 180 || die "Docker Desktop did not start$([ "$PLATFORM" = wsl ] && echo \
          ' (Windows: Settings > Resources > WSL integration must be ON for this distro)'). Open it by hand, then retry."
      fi
      ;;
  esac
  log "Docker is running."
}

desktop_running() {
  case "$PLATFORM" in
    macos) pgrep -f "$DESKTOP_PROC_MAC" >/dev/null ;;
    *)     engine_ok ;;
  esac
}

quit_docker() {
  if [ "$PLATFORM" = linux ]; then
    log "Linux: the Docker engine is a system service, it is left running."
    return 0
  fi
  if ! desktop_running && ! engine_ok; then
    log "Docker Desktop is not running."
    return 0
  fi
  log "Quitting Docker Desktop..."
  docker desktop stop --timeout 60 >/dev/null 2>&1 \
    || docker desktop stop --force --timeout 30 >/dev/null 2>&1 \
    || { [ "$PLATFORM" = macos ] && osascript -e 'quit app "Docker"' >/dev/null 2>&1; } \
    || { [ "$PLATFORM" = wsl ] && powershell.exe -NoProfile -Command \
         "Stop-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue" >/dev/null 2>&1; }
  local t=0
  while desktop_running && [ "$t" -lt 60 ]; do
    sleep 2; t=$((t + 2))
  done
  if [ "$PLATFORM" = macos ] && desktop_running; then
    log "Docker Desktop is still closing: forcing it."
    pkill -f "$DESKTOP_PROC_MAC" 2>/dev/null
  fi
  log "Docker Desktop stopped."
}

# ------------------------------------------------------------------------- project

stack_down() {
  engine_ok || { log "Docker engine not running: no container to stop."; return 0; }
  log "Stopping the project containers..."
  "${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1
  # Leftovers in a broken state (e.g. after Docker was killed) are removed by force.
  docker ps -aq --filter label=com.docker.compose.project=tek5 | xargs docker rm -f >/dev/null 2>&1
  log "Project stopped."
}

stack_running() {
  local n
  n=$("${COMPOSE[@]}" ps --status running -q "${SERVICES[@]}" 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -eq "${#SERVICES[@]}" ]
}

stack_up() {
  if stack_running; then
    log "Project already running."
    return 0
  fi
  log "Starting the project (${SERVICES[*]})..."
  if ! "${COMPOSE[@]}" up -d "${SERVICES[@]}" >"$UP_LOG" 2>&1; then
    log "Start failed (stale containers?): cleaning up and retrying..."
    stack_down
    "${COMPOSE[@]}" up -d "${SERVICES[@]}" >"$UP_LOG" 2>&1 \
      || { cat "$UP_LOG"; die "could not start the project."; }
  fi
  FRESH=1
}

brain_ready() {
  "${COMPOSE[@]}" logs ${SINCE:+--since "$SINCE"} control 2>/dev/null | grep "Brain node ready" >/dev/null
}

wait_brain() {
  local t=0
  log "Waiting for the brain (the workspace is built at each start)..."
  until brain_ready; do
    if [ -z "$("${COMPOSE[@]}" ps --status running -q control 2>/dev/null)" ]; then
      "${COMPOSE[@]}" logs --tail 30 control
      die "the control container stopped (see logs above)."
    fi
    if [ "$t" -ge 180 ]; then
      "${COMPOSE[@]}" logs --tail 30 control
      die "brain not ready after 180 s (see logs above)."
    fi
    sleep 2; t=$((t + 2))
  done
}

ros() { "${COMPOSE[@]}" exec -T control bash -lc "$ROS && $*" </dev/null; }

# ------------------------------------------------------------------------- RViz

show_rviz() {
  if [ "$VIEWER" = vnc ]; then
    log "RViz (browser): $RVIZ_URL"
    return 0
  fi
  # Linux X11: rviz2 runs inside the control container, window on the host display.
  if ros "pgrep -x rviz2" >/dev/null 2>&1; then
    log "RViz already open."
    return 0
  fi
  command -v xhost >/dev/null && xhost +local:docker >/dev/null 2>&1
  "${COMPOSE[@]}" exec -d control bash -lc "$ROS && rviz2 -d /ros2_ws/so101.rviz" \
    && log "RViz window opening (X11, DISPLAY=$DISPLAY). If nothing shows: 'xhost +local:docker' or use --vnc." \
    || log "could not open RViz with X11: try '$0 --vnc'"
}

# ---------------------------------------------------------------------------- main

log "Platform: $PLATFORM, RViz: $VIEWER"
case "$MODE" in
  --stop) stack_down; quit_docker; exit 0 ;;
  --down) stack_down; exit 0 ;;
esac

# ------------------------------------------------------------------------- logging

LOG_PIDS=()
start_logging() {
  mkdir -p ../logs
  LOGFILE="$(cd .. && pwd)/logs/pick_place_$(date +%Y%m%d_%H%M%S).log"
  : >"$LOGFILE"
  log "Logging to $LOGFILE"
  # 1) messages of the nodes (targets, transitions, retries, warnings), from now on
  "${COMPOSE[@]}" logs -f --no-log-prefix --since "$(date -u +%Y-%m-%dT%H:%M:%SZ)" control perception 2>/dev/null \
    | grep --line-buffered -E "\[so101_(brain|perception|driver)\]" \
    | awk '{ sub(/^.*\] \[so101_/, "[so101_"); print "   " $0; fflush() }' | tee -a "$LOGFILE" &
  LOG_PIDS+=($!)
  # 2) live monitor: state, TCP, gripper, perception estimates
  "${COMPOSE[@]}" exec -T control bash -lc "$ROS && exec python3 -m so101_brain.monitor" </dev/null 2>/dev/null \
    | awk '{ print "   " $0; fflush() }' | tee -a "$LOGFILE" &
  LOG_PIDS+=($!)
  sleep 2  # DDS discovery of the monitor
}
stop_logging() {
  [ ${#LOG_PIDS[@]} -eq 0 ] && return 0
  ros "pkill -f '[s]o101_brain.monitor'" >/dev/null 2>&1
  kill "${LOG_PIDS[@]}" 2>/dev/null
  pkill -P $$ -f "logs -f --no-log-prefix" 2>/dev/null
  wait 2>/dev/null
  LOG_PIDS=()
}
trap stop_logging EXIT

# ---------------------------------------------------------------------------- run

run_once() {  # $1 = run index. Sets RUN_OK (0/1), RUN_TIME (s), RUN_ATTEMPTS
  local i=$1 t0 last="" state attempts=1
  RUN_OK=0; RUN_ATTEMPTS=1
  ros "ros2 service call /brain/stop std_srvs/srv/Trigger" >/dev/null 2>&1  # previous run still waiting?
  if [ "$SAME" -eq 0 ]; then
    ros "ros2 service call /sim/reset_object std_srvs/srv/Trigger" >/dev/null 2>&1 \
      && log "run $i/$REPEAT: new random cube position" || log "(no /sim/reset_object: real robot?)"
  fi
  sleep 3  # let the perception see the scene
  ros "ros2 service call /brain/start std_srvs/srv/Trigger" 2>/dev/null | grep "success=True" >/dev/null \
    || die "/brain/start refused (see: docker compose -f docker/docker-compose.yml logs control)"
  log "run $i/$REPEAT: pick & place started"
  t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 180 ]; do
    state=$(ros "timeout 3 ros2 topic echo --once --field data /brain_state 2>/dev/null | grep -v WARNING | head -1" 2>/dev/null)
    if [[ "$state" =~ attempt\ ([0-9]+) ]] && [ "${BASH_REMATCH[1]}" -gt "$attempts" ]; then
      attempts=${BASH_REMATCH[1]}
    fi
    if [ -n "$state" ] && [ "$state" != "$last" ]; then
      [ "$LOG" -eq 0 ] && log "run $i/$REPEAT: $state"   # with --log the monitor prints it
      last="$state"
    fi
    case "$state" in
      DONE*) RUN_OK=1; break ;;
      IDLE*) break ;;
    esac
    sleep 1
  done
  RUN_TIME=$(( $(date +%s) - t0 ))
  RUN_ATTEMPTS=$attempts
  if [ "$RUN_OK" -eq 1 ]; then
    log "run $i/$REPEAT: SUCCESS in ${RUN_TIME}s"
  else
    ros "ros2 service call /brain/stop std_srvs/srv/Trigger" >/dev/null 2>&1
    log "run $i/$REPEAT: FAILED after ${RUN_TIME}s (last state: ${last:-none})"
  fi
}

start_docker
FRESH=0
SINCE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
stack_up
[ "$FRESH" -eq 1 ] || SINCE=""   # already running: the brain was ready before
wait_brain
show_rviz
[ "$LOG" -eq 1 ] && start_logging
[ "$SAME" -eq 1 ] && [ "$REPEAT" -gt 1 ] && log "note: --same with --repeat: the cube is only where it is for run 1"

ok=0; total_time=0; summary=()
for i in $(seq 1 "$REPEAT"); do
  [ "$i" -gt 1 ] && SAME=0
  run_once "$i"
  ok=$((ok + RUN_OK))
  [ "$RUN_OK" -eq 1 ] && total_time=$((total_time + RUN_TIME))
  summary+=("run $i: $([ "$RUN_OK" -eq 1 ] && echo SUCCESS || echo FAILED) ${RUN_TIME}s, grasp attempts: $RUN_ATTEMPTS")
done
stop_logging

log "================ summary ================"
for line in "${summary[@]}"; do log "$line"; done
log "$ok/$REPEAT cubes in the box$([ "$ok" -gt 0 ] && echo ", mean time $((total_time / ok))s")"
[ -n "$LOGFILE" ] && log "full log: $LOGFILE"
log "Run again for more, or '$0 --stop' to quit everything."
[ "$ok" -eq "$REPEAT" ]
