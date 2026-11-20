"""Latency: the end-to-end budget (audio in the air -> the arm moves), how it is measured and printed,
the output-delay (headphones) presets, and the delay line that holds the reactive features back to what
the listener hears.

What is compensated and what is not
  beat clock (moves on the beat)   PREDICTIVE: the clock is read `lead` seconds ahead:
        lead = --latency (display / servo trim) + input device latency + --servo-lead (--real only)
               - output delay (--headphones / --output-delay: the sound reaches the ears that much later)
  reactive features (accents, voice, events)   cannot be predicted: they are only as late as the chain
        window centring + envelope + block + analysis + tick; with a late output (AirPods) they are
        held back by a delay line (StateDelay) so that they coincide with what is heard.
"""
import math
import threading
import time
from collections import deque
from dataclasses import replace

import numpy as np

# Extra acoustic delay of the listening path relative to a wired headphone / the Mac's speakers (s).
# Bluetooth cannot be measured through BlackHole (the capture is before the radio): 150-250 ms typical.
OUTPUT_PRESETS = {"wired": 0.0, "speakers": 0.0, "airpods": 0.18}

# reactive state fields that describe the audio now (held back by the output delay); the clock fields
# (t_ref, theta_ref, rate, off, div, music_bpm, bpm, conf, locked) and the predictions stay current
REACTIVE = ("bass", "mid", "high", "vox_env", "loud_db", "energy", "section", "loud_fast_db", "jump_db",
            "energy_fast", "build_level")


def output_delay(args):
    """Seconds the listener hears the sound later than a wired headphone: --output-delay, else the preset."""
    if getattr(args, "output_delay", None) is not None:
        return float(args.output_delay)
    return OUTPUT_PRESETS.get(getattr(args, "headphones", None) or "wired", 0.0)


def lead_terms(args, source_latency=0.0):
    """The terms of the beat-clock lead, seconds: {"trim", "input", "servo", "output"} (output is negative)."""
    return {"trim": float(args.latency), "input": float(source_latency),
            "servo": float(args.servo_lead) if getattr(args, "real", False) else 0.0,
            "output": -output_delay(args)}


class StateDelay:
    """Holds the reactive fields of the BeatState back `delay` seconds (stream time) and leaves the clock
    and the predictions current."""

    def __init__(self, delay):
        self.delay = max(float(delay), 0.0)
        self.h = deque()

    def __call__(self, ts, st):
        if self.delay <= 0.0:
            return st
        self.h.append((ts, st))
        while len(self.h) > 1 and self.h[1][0] <= ts - self.delay:
            self.h.popleft()
        old = self.h[0][1]  # the newest state at least `delay` old (the oldest we have while starting)
        return replace(st, **{k: getattr(old, k) for k in REACTIVE})


class RunStats:
    """Measurements of the real-time run (thread-safe appends)."""

    def __init__(self):
        self.an_lag = deque(maxlen=8000)   # s: newest sample's delivery into the ring -> its analysis done
        self.late = deque(maxlen=20000)    # s: control tick lateness
        self.print_t = None


def bench_analyzer(make, sr, hop, seconds=2.0):
    """Cost of the analysis on this machine: ms per fast hop (mean, p99, max) and per tempo frame."""
    an = make()
    rng = np.random.default_rng(0)
    # a click-like signal (not silence: silence skips parts of the pipeline): 120 bpm kick + noise
    n = int(seconds * sr)
    t = np.arange(n) / sr
    y = (0.05 * rng.standard_normal(n) + 0.6 * np.exp(-((t % 0.5) * 30.0)) * np.sin(2 * np.pi * 60 * t)).astype(np.float32)
    fh = an.fast_hop
    costs, tcost = [], []
    for k in range(n // fh):
        t0 = time.perf_counter()
        f0 = an.frames
        an.process(y[k * fh:(k + 1) * fh])
        c = time.perf_counter() - t0
        costs.append(c)
        if an.frames != f0:
            tcost.append(c)
    c = np.array(costs[len(costs) // 4:]) * 1000.0
    tc = np.array(tcost[len(tcost) // 4:]) * 1000.0
    return {"mean": float(c.mean()), "p99": float(np.percentile(c, 99)), "max": float(c.max()),
            "tempo_mean": float(tc.mean()) if len(tc) else 0.0, "hop_ms": fh / sr * 1000.0,
            "load": float(c.mean() / (fh / sr * 1000.0))}


def budget_rows(args, an, source, tick_s, bench=None, twin=None):
    """[(stage, typical ms, worst ms, note)] for the two chains (reactive features, beat clock)."""
    rows = []
    sr = an.sr
    lead = lead_terms(args, getattr(source, "latency", 0.0))
    block = getattr(source, "block_s", 0.0) * 1000.0
    rows.append(("-- reactive chain: a kick in the air -> the arm accents it", None, None, ""))
    rows.append(("input device (known, compensated)", lead["input"] * 1000, lead["input"] * 1000,
                 "PortAudio latency minus the block, stream_time() covers the block"))
    rows.append(("capture block", block / 2, block, f"{getattr(source, 'fast_hop', an.fast_hop)} samples"))
    if bench:
        rows.append(("analysis compute per fast hop", bench["mean"], bench["p99"],
                     f"{100 * bench['load']:.0f} % of the {bench['hop_ms']:.1f} ms hop; queue measured live"))
    rows.append(("band window centring (accents)", an.n_short / 2 / sr * 1000, an.n_short / 2 / sr * 1000,
                 f"{an.n_short} samples Hann (tempo frames: {an.n_fft} + onset_shift, compensated in the clock)"))
    rows.append(("band attack smoothing", args.band_attack * 1000, 3 * args.band_attack * 1000,
                 "release 150 ms: only the decay"))
    rows.append(("loudness for events (jump_db)", args.fast_loud_tau * 1000, 3 * args.fast_loud_tau * 1000,
                 "energy / sections: tau %.1f s + %.1f s dwell" % (args.loud_tau, args.section_dwell)))
    rows.append(("control tick", tick_s * 500, tick_s * 1000, f"{1 / tick_s:.0f} Hz"))
    if twin is not None:
        rows.append(("twin render + MJPEG (server side)", 1000 * (0.5 / twin.fps + 0.012), 1000 * (1.0 / twin.fps + 0.03),
                     f"{twin.fps:.0f} fps render + JPEG + HTTP poll; browser buffering not included"))
    if getattr(args, "real", False):
        rows.append(("servo response", args.servo_lead * 1000, args.servo_lead * 1000,
                     "compensated on the beat clock by --servo-lead (not on reactive accents)"))
    rows.append(("-- beat clock: predicted, so all of it is compensated, what is left is the phase error", None, None, ""))
    for k, name in (("trim", "--latency (display / servo trim)"), ("input", "input device"),
                    ("servo", "--servo-lead (--real)"), ("output", "output delay (--headphones)")):
        rows.append((f"lead: {name}", lead[k] * 1000, lead[k] * 1000, ""))
    rows.append(("lead total", sum(lead.values()) * 1000, sum(lead.values()) * 1000, "beat clock read this far ahead"))
    return rows


def print_budget(rows, title="latency budget"):
    print(f"{title} (ms: typical / worst)")
    for stage, typ, worst, note in rows:
        if typ is None:
            print(f"  {stage}")
        else:
            print(f"    {stage:<38s} {typ:7.1f} / {worst:6.1f}  {note}")


def reactive_total(rows):
    """Sum of the typical / worst of the reactive-chain rows that add up (not the clock lead)."""
    typ = worst = 0.0
    for stage, t, w, _ in rows:
        if t is None and stage.startswith("-- beat clock"):
            break
        if t is not None and not stage.startswith("loudness for events"):
            typ += t
            worst += w
    return typ, worst
