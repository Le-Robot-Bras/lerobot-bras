"""dance_live.py - the SO-ARM101 dances in REAL TIME to whatever music plays on the Mac.

    UV="uv run --no-project --python 3.12 --with sounddevice --with numpy --with mujoco --with opencv-python-headless"
    $UV tools/dance_live.py --list-devices                 # audio inputs
    $UV tools/dance_live.py                                # live sim twin, input "BlackHole" (http://localhost:8011)
    $UV tools/dance_live.py --file song.mp3                # test without BlackHole: the file is played (system player)
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
import argparse
import sys

from .app import run_fast, run_live
from .audio_io import list_devices
from .latency import OUTPUT_PRESETS
from .moves import STYLES
from .safety import check_moves


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sim", action="store_true", help="live MuJoCo twin only (default)")
    g.add_argument("--real", action="store_true", help="the real arm through real_arm_server (+ the twin)")
    p.add_argument("--device", default="BlackHole", help="input device name substring (default BlackHole)")
    p.add_argument("--list-devices", action="store_true", help="print the audio devices and exit")
    p.add_argument("--file", help="test source: this song, fed in real time through the same analysis")
    p.add_argument("--audio-out", action=argparse.BooleanOptionalAction, default=True,
                   help="--file: play it with the system player (afplay / ffplay / paplay / pw-play; --no-audio-out: silent)")
    p.add_argument("--audio-latency", type=float, default=0.1, help="--file: the system player's start delay (s)")
    p.add_argument("--fast", action="store_true", help="--file: headless, faster than real time (tests)")
    p.add_argument("--duration", type=float, help="stop after this many seconds of audio")
    p.add_argument("--log", help="save the control log (npz) here")
    g = p.add_argument_group("latency (the budget is printed at startup)")
    g.add_argument("--latency", type=float, default=0.05,
                   help="s the beat clock runs ahead (display / output trim); the input device latency is "
                        "added, and with --real --servo-lead, minus the output delay (--headphones)")
    g.add_argument("--servo-lead", type=float, default=0.08,
                   help="--real only: the joint targets are sent this far ahead in beat time (servo response, s)")
    g.add_argument("--headphones", choices=sorted(OUTPUT_PRESETS),
                   help="what you listen with: the beat clock (and the reactive features) are held back by the "
                        "extra acoustic delay of the output. wired / speakers: 0 (default), airpods: %.0f ms "
                        "(Bluetooth adds 150-250 ms, which BlackHole cannot measure: fine-tune with --output-delay)"
                        % (1000 * OUTPUT_PRESETS["airpods"]))
    g.add_argument("--output-delay", type=float, default=None,
                   help="s the sound reaches your ears later than a wired headphone (overrides --headphones); "
                        "tools/latency_calibrate.py measures the part BlackHole can see")
    g.add_argument("--tick-hz", type=float, default=50.0, help="control loop rate (the arm gets a command per tick)")
    g.add_argument("--hop-div", type=int, default=4,
                   help="fast analysis hops per tempo frame (~5 ms hops: accents, loudness, drop anticipation); "
                        "the tempo tracker still gets its ~23 ms frames. 1 = old single rate")
    g.add_argument("--accent-ms", type=float, default=46.0,
                   help="window of the band levels (bass / mid / high accents), ms; the tempo frames keep ~93 ms")
    g.add_argument("--band-attack", type=float, default=0.003, help="band envelope attack time constant (s)")
    g.add_argument("--band-release", type=float, default=0.15, help="band envelope release time constant (s)")
    g.add_argument("--loud-tau", type=float, default=3.0, help="loudness behind energy / sections (s)")
    g.add_argument("--fast-loud-tau", type=float, default=0.10, help="fast loudness for jump_db / energy_fast (s)")
    g.add_argument("--section-dwell", type=float, default=1.5, help="s a calm / normal / drop change must persist")
    g.add_argument("--drop-max", type=float, default=45.0,
                   help="a drop lasting longer than this many s is the song's normal level: back to normal for 20 s")
    g.add_argument("--drop-dwell", type=float, default=2.5, help="s the drop conditions must hold to enter 'drop'")
    g.add_argument("--drop-rank", type=float, default=0.80,
                   help="drop needs the loudness in the top (1 - this) of THIS song's own distribution")
    g.add_argument("--drop-over-db", type=float, default=1.0,
                   help="drop also needs this many dB above the song's median loudness (and more onsets / bass)")
    g.add_argument("--song-dist", type=float, default=3.5,
                   help="a new song without a gap = 2 of: spectral shape moved by this many dB, level by 3 dB, tempo by 6 %%")
    g.add_argument("--song-gap", type=float, default=1.2, help="s of silence that separate two songs")
    g.add_argument("--anticipate", action=argparse.BooleanOptionalAction, default=True,
                   help="build-up detection and drop prediction (BeatState.build_level / drop_eta)")
    g.add_argument("--legacy-analysis", action="store_true",
                   help="the analysis as before the latency work (one 23 ms hop, 93 ms band window, 10 ms attack)")
    p.add_argument("--amp", type=float, default=1.0, help="overall amplitude scale, <= 1")
    p.add_argument("--vmax", type=float, default=None,
                   help="joint speed limit (rad/s); default 3.5 in --sim, 1.5 with --real (safer for the "
                        "real servos: raise it there only on purpose)")
    p.add_argument("--amax", type=float, default=None,
                   help="joint acceleration limit (rad/s^2); default 120 in --sim, 12 with --real")
    p.add_argument("--margin", type=float, default=0.15, help="distance kept from the joint limits (rad)")
    p.add_argument("--max-dance-bpm", type=float, default=None,
                   help="faster songs are danced half time (default: per style, 110-140, 124 without a style)")
    p.add_argument("--guard", action=argparse.BooleanOptionalAction, default=True,
                   help="MuJoCo contact check of every commanded pose (hold on contact)")
    p.add_argument("--freeze-time", type=float, default=2.0, help="s held after a contact")
    p.add_argument("--seed", type=int, default=0, help="move choice randomness")
    p.add_argument("--genre", action=argparse.BooleanOptionalAction, default=True,
                   help="detect the music genre live (YAMNet via mediapipe) and dance in its style")
    p.add_argument("--style", choices=list(STYLES), help="force a dance style (no genre detection needed)")
    p.add_argument("--gripper", action=argparse.BooleanOptionalAction, default=None,
                   help="the gripper opens with the singing voice like a mouth (default: on in --sim, off with "
                        "--real where the gripper calibration is fragile)")
    p.add_argument("--vocal-model", action=argparse.BooleanOptionalAction, default=True,
                   help="vocal detection with Demucs htdemucs source separation (needs --with demucs --with "
                        "torchaudio; falls back to YAMNet + DSP cues without it)")
    p.add_argument("--percussive-onset", action=argparse.BooleanOptionalAction, default=False,
                   help="beat tracking on the percussive part of the spectrum only (harmonic/percussive "
                        "split). Off by default: on the 9-song benchmark it did not help (Ripple tempo 81->71 %%)")
    p.add_argument("--check-moves", action="store_true",
                   help="run every move through dance.py's limit/speed/collision checks at 130 bpm and exit")
    p.add_argument("--view", action=argparse.BooleanOptionalAction, default=True, help="live twin in the browser")
    p.add_argument("--view-port", type=int, default=8011)
    p.add_argument("--ros-bridge", action="store_true",
                   help="serve the dance state for ROS 2 / RViz (ros2 run so101_driver dance_bridge in Docker)")
    p.add_argument("--ros-bridge-host", default="127.0.0.1")
    p.add_argument("--ros-bridge-port", type=int, default=5600)
    p.add_argument("--view-wait", type=float, default=2.0, help="--sim: s before the audio starts")
    p.add_argument("--headless", action="store_true", help="--sim without the twin (status lines only)")
    p.add_argument("--verbose", action="store_true", help="--fast: a status line every second")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--approach-speed", type=float, default=0.4, help="rad/s to reach / leave the base pose")
    p.add_argument("--max-lag", type=float, default=0.35, help="abort if a joint lags its target this much (rad)")
    p.add_argument("--settle-time", type=float, default=5.0)
    p.add_argument("--settle-tol", type=float, default=0.08)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.list_devices:
        list_devices()
        return None
    if args.check_moves:
        sys.exit(0 if check_moves() else 1)
    args.amp = min(args.amp, 1.0)
    if args.vmax is None:
        args.vmax = 1.5 if args.real else 3.5
    if args.amax is None:
        args.amax = 12.0 if args.real else 120.0
    if args.gripper is None:
        args.gripper = not args.real
    args.max_dance_bpm_auto = args.max_dance_bpm is None
    if args.max_dance_bpm is None:
        args.max_dance_bpm = 124.0
    if args.fast:
        if not args.file:
            raise SystemExit("--fast needs --file")
        ctl = run_fast(args)
    else:
        ctl = run_live(args)
    if ctl is not None and args.log:
        ctl.save_log(args.log)
    return ctl


def dance_live(device="BlackHole", mode="sim", file=None, **options):
    """Dance live to the Mac's sound (input `device`), or to `file` (played + analysed in real time).
    mode: "sim" (live twin on http://localhost:8011) or "real" (real_arm_server --allow-motion).
    options: any CLI flag, e.g. amp=0.7, vmax=1.2, latency=0.12, duration=60, audio_out=False,
    fast=True, log="out.npz". Returns the Controller (its .log has the whole run)."""
    argv = ["--" + mode, "--device", device] + (["--file", str(file)] if file else [])
    for k, v in options.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v is False:
            if k in ("audio_out", "guard", "view", "genre", "gripper", "percussive_onset", "vocal_model", "anticipate"):
                argv.append("--no-" + k.replace("_", "-"))
        elif v is not None:
            argv += [flag, str(v)]
    return main(argv)
