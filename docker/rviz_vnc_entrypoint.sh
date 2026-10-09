#!/usr/bin/env bash
# Virtual X server + VNC + noVNC, then RViz once the workspace is built.
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export QT_QPA_PLATFORM=xcb
export QT_X11_NO_MITSHM=1
export MESA_GL_VERSION_OVERRIDE=3.3
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"

Xvfb :99 -screen 0 1600x900x24 +extension GLX +render -noreset &
until xdpyinfo -display :99 >/dev/null 2>&1; do sleep 0.2; done

openbox &
x11vnc -display :99 -forever -shared -nopw -quiet -rfbport 5900 &
websockify --web /usr/share/novnc 6080 localhost:5900 &

source /opt/ros/jazzy/setup.bash
until [ -f /ros2_ws/install/setup.bash ]; do echo "waiting for build (control)..."; sleep 1; done
source /ros2_ws/install/setup.bash

echo "RViz available at http://localhost:6080/vnc.html"
# Restart RViz if it is closed, so the window can always be recovered.
while true; do
  rviz2 -d /ros2_ws/so101.rviz || true
  sleep 2
done
