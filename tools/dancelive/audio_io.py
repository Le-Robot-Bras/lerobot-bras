"""Audio input: ring buffer, live device source, real-time file source."""
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

# ================================================================ audio sources
class RingBuffer:
    """Single producer / single consumer float32 ring. Keeps the total sample count (stream clock)."""

    def __init__(self, size):
        self.buf = np.zeros(size, np.float32)
        self.size = size
        self.w = 0  # samples written since the start (the stream clock)
        self.r = 0
        self.t_write = None  # monotonic time of the last write
        self.overruns = 0
        self.closed = False
        self.cv = threading.Condition()
        self.stamps = deque(maxlen=512)  # (samples written so far, monotonic time of that write)
        self.sbuf = np.zeros((2, size), np.float32)  # the two channels of the same samples (stereo for Demucs)

    def write(self, x, st=None):
        """x: mono samples (n,); st: the same samples as (2, n) stereo (default: the mono duplicated)."""
        x = np.asarray(x, np.float32)
        st = np.stack([x, x]) if st is None else np.asarray(st, np.float32)
        with self.cv:
            n = len(x)
            y = x[-self.size:]
            ys = st[:, -self.size:]
            i = (self.w + n - len(y)) % self.size
            k = min(len(y), self.size - i)
            self.buf[i:i + k] = y[:k]
            self.buf[:len(y) - k] = y[k:]
            self.sbuf[:, i:i + k] = ys[:, :k]
            self.sbuf[:, :len(y) - k] = ys[:, k:]
            self.w += n
            if self.w - self.r > self.size:  # the analysis fell behind: drop the oldest audio
                self.overruns += 1
                self.r = self.w - self.size
            self.t_write = time.monotonic()
            self.stamps.append((self.w, self.t_write))
            self.cv.notify_all()

    def age(self, idx, now=None):
        """Seconds since the audio up to sample count `idx` was delivered into the ring (None: unknown)."""
        now = time.monotonic() if now is None else now
        with self.cv:
            for w, t in self.stamps:
                if w >= idx:
                    return now - t
        return None

    def read(self, n, timeout=0.5, stereo=False):
        """n samples, blocking; None on timeout or when closed and drained. stereo=True: (mono, (2, n))."""
        with self.cv:
            t_end = time.monotonic() + timeout
            while self.w - self.r < n:
                if self.closed or not self.cv.wait(max(0.0, t_end - time.monotonic())):
                    if self.w - self.r < n:
                        return None
            i = self.r % self.size
            k = min(n, self.size - i)
            out = np.concatenate([self.buf[i:i + k], self.buf[:n - k]])
            if stereo:
                out = (out, np.concatenate([self.sbuf[:, i:i + k], self.sbuf[:, :n - k]], axis=1))
            self.r += n
            return out

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()


class Source:
    """Common part: sr, ring, stream_time(now) = seconds of audio delivered so far, extrapolated."""
    sr = 22050
    latency = 0.0  # input latency (s) behind real time that stream_time() does NOT already cover
    block_s = 0.0  # audio delivered per ring write (s)
    done = False
    hop = 512       # tempo frame (samples): LiveAnalyzer(hop=...)
    fast_hop = 128  # samples per read / fast analysis hop

    def stream_time(self, now=None):
        rb = self.ring
        with rb.cv:
            w, tw = rb.w, rb.t_write
        if tw is None:
            return 0.0
        now = time.monotonic() if now is None else now
        return w / self.sr + min(max(now - tw, 0.0), 0.1)


def list_devices():
    import sounddevice as sd
    print(sd.query_devices())


def find_input_device(name):
    """(index, info) of the first input device whose name contains `name`, else the default input."""
    import sounddevice as sd
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0 and name.lower() in d["name"].lower():
            return i, d
    inputs = [f"  [{i}] {d['name']}" for i, d in enumerate(devs) if d["max_input_channels"] > 0]
    try:
        idx = sd.default.device[0]
        d = sd.query_devices(idx, "input")
        idx = d["index"] if "index" in d else idx
    except Exception:
        raise SystemExit(f"no input device matches {name!r} and there is no default input:\n" + "\n".join(inputs))
    print(f"WARNING: no input device matches {name!r}. Inputs:\n" + "\n".join(inputs)
          + f"\n-> falling back to the default input [{idx}] {d['name']} (a microphone hears the room, not the"
          + " Mac's sound). For loopback: brew install blackhole-2ch, then Audio MIDI Setup -> Multi-Output"
          " Device = speakers + BlackHole 2ch, selected as the output."
          + ("\nOn Linux (PipeWire / PulseAudio) capture the monitor of your output instead: run with "
             "`--device pulse` and PULSE_SOURCE=<sink name>.monitor (list sinks: pactl list short sinks)."
             if sys.platform.startswith("linux") else ""))
    return idx, d


class DeviceSource(Source):
    """Live input. Blocks of `fast_hop` samples (5.3 ms at 48 kHz) with PortAudio's 'low' latency: the
    ring and the analysis then see the audio ~15 ms sooner than with whole 21 ms tempo-hop blocks."""

    def __init__(self, name="BlackHole", hop_div=4):
        import sounddevice as sd
        idx, info = find_input_device(name)
        self.name = info["name"]
        self.sr = int(info["default_samplerate"])
        self.hop = 512 if self.sr < 32000 else 1024
        d = max(1, int(hop_div))
        while self.hop % d:
            d -= 1
        self.fast_hop = self.hop // d
        self.block_s = self.fast_hop / self.sr
        self.ring = RingBuffer(self.sr * 4)
        self.status_errors = 0
        self.cb_times = deque(maxlen=4096)  # monotonic time of each callback
        self.cb_ages = deque(maxlen=4096)   # PortAudio: capture time of the block's last sample -> callback
        self.device_latency = float(info.get("default_low_input_latency", 0.0))

        def callback(indata, frames, time_info, status):
            if status:
                self.status_errors += 1
            now = time.monotonic()
            self.cb_times.append(now)
            try:  # age of the newest sample when we get it (device + safety offset; ~0 for BlackHole)
                age = time_info.currentTime - time_info.inputBufferAdcTime - frames / self.sr
                if 0.0 <= age < 0.5:
                    self.cb_ages.append(age)
            except Exception:
                pass
            # mono mix for the analysis, both channels for Demucs (a mono device: the channel twice)
            self.ring.write(indata.mean(axis=1), indata[:, :2].T if indata.shape[1] >= 2 else None)

        self.stream = sd.InputStream(device=idx, channels=min(2, int(info["max_input_channels"])),
                                     samplerate=self.sr, blocksize=self.fast_hop, dtype="float32",
                                     latency="low", callback=callback)
        # PortAudio's latency counts the block itself (BlackHole 1024 @ 48 kHz: 21.3 ms = one block), but
        # stream_time() already extrapolates through the block: only what is left is real input latency
        self.stream_latency = float(self.stream.latency)
        self.latency = max(self.stream_latency - self.block_s, 0.0)

    def start(self):
        self.stream.start()
        print(f"audio input: {self.name} @ {self.sr} Hz, {self.fast_hop}-sample blocks ({self.block_s * 1000:.1f} ms), "
              f"PortAudio input latency {self.stream_latency * 1000:.1f} ms (device part {self.latency * 1000:.1f} ms)")

    def callback_stats(self):
        """(mean period ms, max period ms, median capture age ms or None) of the audio callbacks so far."""
        t = np.array(self.cb_times)
        ages = np.array(self.cb_ages)
        if len(t) < 10:
            return None
        d = np.diff(t) * 1000.0
        return float(d.mean()), float(d.max()), (float(np.median(ages) * 1000.0) if len(ages) else None)

    def close(self):
        self.stream.stop()
        self.stream.close()
        self.ring.close()


def decode_file(path, sr=22050, channels=1):
    """float32 samples of any audio file (ffmpeg, else librosa): (n,) mono, or (2, n) with channels=2."""
    path = str(Path(path).expanduser())
    if shutil.which("ffmpeg"):
        raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-f", "f32le",
                              "-ac", str(channels), "-ar", str(sr), "-"], check=True, capture_output=True).stdout
        x = np.frombuffer(raw, np.float32).copy()
        return x if channels == 1 else x.reshape(-1, channels).T.copy()
    import librosa
    y = librosa.load(path, sr=sr, mono=(channels == 1))[0].astype(np.float32)
    return y if channels == 1 else (y if y.ndim == 2 else np.stack([y, y]))


def decode_stereo_like(path, sr, n):
    """(2, n) stereo decode of `path` (cut / padded to n samples), for the mono decode of length n."""
    s = decode_file(path, sr, channels=2)
    if s.shape[1] < n:
        s = np.pad(s, ((0, 0), (0, n - s.shape[1])))
    return s[:, :n]


def player_command(path):
    """(argv, temp wav or None) of the system player for `path`: afplay on macOS; elsewhere the first of ffplay,
    paplay, pw-play (the last two need a wav: decoded with ffmpeg to a temp file). (None, None) if none."""
    path = str(path)
    if sys.platform == "darwin" and shutil.which("afplay"):
        return ["afplay", path], None
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path], None
    for exe in ("paplay", "pw-play"):
        if shutil.which(exe):
            if path.lower().endswith(".wav"):
                return [exe, path], None
            if not shutil.which("ffmpeg"):
                continue
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="dancelive_")
            os.close(fd)
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path, tmp], check=True)
            return [exe, tmp], tmp
    return None, None


class FileSource(Source):
    """A song fed in real time (wall clock) through the same ring buffer as the live input; played
    with the system player (afplay / ffplay / paplay / pw-play) unless play=False. Sample k is released at t0 + k/sr, t0 = player start + audio_latency."""

    def __init__(self, path, sr=22050, play=True, audio_latency=0.1, hop_div=4):
        self.path = Path(path).expanduser()
        self.sr, self.hop = sr, 512
        d = max(1, int(hop_div))
        while self.hop % d:
            d -= 1
        self.fast_hop = self.hop // d
        self.block_s = 0.005  # the feeder wakes every 5 ms
        self.y = decode_file(self.path, sr)
        self.ys = decode_stereo_like(self.path, sr, len(self.y))  # same samples, both channels
        self.ring = RingBuffer(sr * 4)
        self.play, self.audio_latency = play, audio_latency
        self.proc = None
        self.tmp_wav = None
        self._stop = threading.Event()

    def start(self):
        if self.play:
            argv, self.tmp_wav = player_command(self.path)
            if argv is None:
                print("WARNING: no audio player found (afplay / ffplay / paplay / pw-play): continuing silently, "
                      "as with --no-audio-out")
                self.play = False
            else:
                self.proc = subprocess.Popen(argv)
        t0 = time.monotonic() + (self.audio_latency if self.play else 0.0)
        print(f"audio file: {self.path.name} ({len(self.y) / self.sr:.0f} s)"
              + (", playing it" if self.play else ", silent (--no-audio-out)"))
        threading.Thread(target=self._feed, args=(t0,), daemon=True, name="file-feeder").start()

    def _feed(self, t0):
        k, n = 0, len(self.y)
        while k < n and not self._stop.is_set():
            due = int((time.monotonic() - t0) * self.sr)
            if due > k:
                due = min(due, n)
                self.ring.write(self.y[k:due], self.ys[:, k:due])
                k = due
            time.sleep(0.005)
        self.done = True
        self.ring.close()

    def close(self):
        self._stop.set()
        self.ring.close()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        if self.tmp_wav:
            try:
                os.unlink(self.tmp_wav)
            except OSError:
                pass
            self.tmp_wav = None


def make_source(args):
    if args.file:
        return FileSource(args.file, sr=22050, play=args.audio_out, audio_latency=args.audio_latency,
                          hop_div=args.hop_div)
    return DeviceSource(args.device, hop_div=args.hop_div)
