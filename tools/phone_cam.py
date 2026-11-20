#!/usr/bin/env python3
"""phone_cam.py - serves an Android phone's camera, plugged in USB, as an MJPEG stream.

    python3 tools/phone_cam.py                      # -> http://localhost:8090/video
    python3 tools/phone_cam.py --camera-id 2 --size 1920x1080 --fps 15

Why: no Wi-Fi camera app needed, no macOS camera permission. scrcpy (>= 2.2, `brew install
scrcpy`, which also brings adb) reads the phone camera over USB/adb; ffmpeg decodes its H.264 and
re-encodes JPEG frames; this script serves them:
    /video     MJPEG stream (cv2.VideoCapture("http://localhost:8090/video"), browser)
    /shot.jpg  latest frame
The Docker containers reach it as http://host.docker.internal:8090/video (CAMERA_DEVICE).

Phone: USB debugging enabled and this Mac authorised (`adb devices` says "device"), and no other
app using the camera (close the Camera app: Android refuses a second client).
`scrcpy --list-cameras` lists the camera ids / sizes. Restarts scrcpy if the phone drops.
Needs scrcpy and ffmpeg on the PATH (Homebrew), Python stdlib only.
"""
import argparse
import http.server
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path


class Camera:
    def __init__(self, args):
        self.args = args
        self.cond = threading.Condition()
        self.jpeg, self.seq, self.t_last = None, 0, 0.0
        self.fifo = Path(tempfile.mkdtemp(prefix="phone_cam_")) / "cam.mkv"
        os.mkfifo(self.fifo)

    def _session(self):
        a = self.args
        scr = subprocess.Popen(
            ["scrcpy", "--video-source=camera", f"--camera-id={a.camera_id}", f"--camera-size={a.size}",
             f"--camera-fps={a.fps}", "--no-audio", "--no-playback", "--no-window",
             f"--record={self.fifo}", "--record-format=mkv"] + (["-s", a.serial] if a.serial else []),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        ff = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-fflags", "nobuffer", "-f", "matroska",
             "-i", str(self.fifo), "-pix_fmt", "yuvj420p", "-q:v", str(a.quality),
             "-f", "image2pipe", "-c:v", "mjpeg", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        threading.Thread(target=lambda: [print("scrcpy:", l.rstrip()) for l in scr.stderr
                                         if "ERROR" in l or "WARN" in l or "camera" in l.lower()],
                         daemon=True).start()
        buf = b""
        try:
            while chunk := ff.stdout.read(65536):
                buf += chunk
                while (s := buf.find(b"\xff\xd8")) >= 0 and (e := buf.find(b"\xff\xd9", s + 2)) >= 0:
                    with self.cond:
                        self.jpeg, self.seq, self.t_last = buf[s:e + 2], self.seq + 1, time.time()
                        self.cond.notify_all()
                    buf = buf[e + 2:]
        finally:
            for p in (ff, scr):
                p.terminate()
                try:
                    p.wait(3)
                except subprocess.TimeoutExpired:
                    p.kill()

    def run(self):
        while True:
            self._session()
            print("stream ended (phone unplugged, camera busy?), restarting in 2 s")
            time.sleep(2)


def make_handler(cam):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/video"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with cam.cond:
                            cam.cond.wait_for(lambda: cam.seq != seen, timeout=5)
                            jpg, seen = cam.jpeg, cam.seq
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif self.path.startswith("/shot.jpg"):
                with cam.cond:
                    jpg = cam.jpeg
                if not jpg:
                    self.send_error(503, "no frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b'<body style="background:#111;margin:0"><img src="video" style="max-width:100%">')
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-id", default="0", help="see `scrcpy --list-cameras` (0 = main back camera)")
    p.add_argument("--size", default="1280x720", help="camera size WxH (setup.md expects 1280x720)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--quality", type=int, default=4, help="JPEG quality, 2 (best) .. 31")
    p.add_argument("--serial", help="adb serial, if several devices are plugged in")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    args = p.parse_args()
    for tool in ("scrcpy", "ffmpeg", "adb"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found: brew install scrcpy ffmpeg")
    cam = Camera(args)
    threading.Thread(target=cam.run, daemon=True).start()
    srv = http.server.ThreadingHTTPServer((args.host, args.port), make_handler(cam))
    print(f"http://localhost:{args.port}/video  (Docker: http://host.docker.internal:{args.port}/video)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
