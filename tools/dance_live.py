#!/usr/bin/env python3
"""dance_live.py - the SO-ARM101 dances in REAL TIME to whatever music plays on the Mac.

    UV="uv run --no-project --python 3.12 --with sounddevice --with numpy --with mujoco --with opencv-python-headless"
    $UV tools/dance_live.py --list-devices                 # audio inputs
    $UV tools/dance_live.py                                # live sim twin, input "BlackHole" (http://localhost:8011)
    $UV tools/dance_live.py --file song.mp3                # test without BlackHole: the file is played (afplay)
                                                           # and fed in real time through the same analysis
    $UV tools/dance_live.py --real                         # real arm (tools/real_arm_server.py --allow-motion)
    $UV tools/dance_live.py --file song.mp3 --fast --log out.npz   # headless, faster than real time, logged

From Python:  from dance_live import dance_live
    dance_live()                                   # BlackHole -> live sim view
    dance_live(file="song.mp3", duration=60)       # test source
    dance_live(mode="real", amp=0.6, latency=0.12)

Capturing the Mac's sound: install BlackHole (brew install blackhole-2ch), then in Audio MIDI Setup
create a Multi-Output Device = your speakers/headphones + BlackHole 2ch and select it as the sound
output. Everything that plays is then heard AND captured from the "BlackHole 2ch" input.

Pipeline (all causal, nothing looks ahead):
  audio callback -> ring buffer -> analysis thread (one FFT per hop of ~23 ms):
    onset strength  spectral flux of ~40 log-spaced band levels (dB), minus its 0.4 s running mean
    bands           bass 20-150 Hz, mids 150-2000, highs 2000-8000: dB envelopes (fast attack, 0.2 s
                    release) normalized to a peak that decays 1 dB/s (0 = 18 dB under it, 1 = at it)
    loudness        3 s envelope vs. its own slowly decaying peak -> energy 0..1 ->
                    section silence / calm / normal / drop (hysteresis + 1.5 s dwell)
    tempo           autocorrelation of the last 8 s of onset strength, every 0.25 s, scored over
                    70-180 bpm (lag + 2*lag), smoothed over ~2.5 s, hysteresis on switches;
                    above --max-dance-bpm the arm dances half time
    beat clock      a phase-locked loop: a continuous beat counter runs at the tempo; every hop the
                    circular mean of the onset strength of the last 4 s, taken at the clock's own
                    phase, says where the onsets fall relative to the clock's beats; the phase is
                    nudged toward it (and a tiny integral term trims the rate). It PREDICTS beats:
                    the motion reads the clock at now + --latency.
  control loop (50 Hz): one move of tools/dance.py per 8-beat phrase (tier from the section, as in
  dance_music.TIERS, never the same twice), faded in/out over the first/last beat of the phrase,
  amplitude 0.4..1 from the loudness, small bass bounce on the beat and highs shimmer on the wrist,
  gentle sway only while the tempo is uncertain, back to BASE when the music stops. Safety on every
  command: velocity + acceleration limits (--vmax, --amax), joint limits with a margin, amplitude
  ceiling (moves + accents never exceed the amplitude the offline checks validated, and the moves
  shrink to stay under --vmax at fast tempos), and a MuJoCo self/table contact check of the next
  pose before it is sent (hold on contact).
"""
# The implementation lives in the tools/dancelive/ package (one module per concern, see its
# __init__); this file is the CLI entry point and re-exports everything, so
# `python tools/dance_live.py ...`, `from dance_live import dance_live` and `import dance_live as dl`
# keep working unchanged.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dancelive import *  # noqa: E402,F401,F403
from dancelive import _derivs, _ema, _style_twin_class  # noqa: E402,F401
from dancelive.cli import build_parser, dance_live, main  # noqa: E402,F401

if __name__ == "__main__":
    main()
