#!/usr/bin/env python3
"""latency_calibrate.py - measure how much later (or earlier) the BlackHole capture of what the Mac plays
is than the moment the sound leaves the output device, so that tools/dance_live.py can move the arm
with what you HEAR (--output-delay / --headphones).

    UV="uv run --no-project --python 3.12 --with sounddevice --with numpy"
    $UV tools/latency_calibrate.py                         # play 12 clicks on the default output, record BlackHole
    $UV tools/latency_calibrate.py --output "Dance Output" --input BlackHole
    $UV tools/latency_calibrate.py --selftest              # synthetic recording, no audio device touched
    $UV tools/latency_calibrate.py --analyze rec.npz       # re-analyse a saved recording (--save rec.npz)

STOP dance_live.py first: it reads BlackHole, and a click train through "Dance Output" would be danced to.
The tool PLAYS SOUND (a short chirp click, 12 times, about 10 s) through the default output.

What is measured
  Two PortAudio streams share one host clock: the output stream tells at which host time each click sample
  reaches the DAC of the output device (outputBufferDacTime + the device's own latency), the input stream
  at which host time each captured sample was recorded (inputBufferAdcTime). The click is found in the
  capture by cross-correlation (sub-sample). L_bh = capture time - DAC time of the click, per click, then
  median and spread.
      L_bh > 0   BlackHole delivers the audio later than it leaves the output  -> the arm would lag: the
                 beat clock must lead more
      L_bh < 0   BlackHole is ahead of the output (CoreAudio's multi-output device delays the sound of its
                 slower sub-devices to keep them in sync)
  The suggested dance_live.py option is   --output-delay (-L_bh)   in seconds (the sound is heard that
  much later than the analysis sees it; negative = earlier).

What cannot be measured here
  Bluetooth headphones (AirPods): the radio + codec delay (150-250 ms, AAC / SBC) happens AFTER the point the
  Mac can time-stamp, and BlackHole captures before it. CoreAudio reports part of it as the device's
  latency (already inside the DAC time), the rest is invisible. Add the invisible part by ear:
  --headphones airpods (adds 180 ms) then --output-delay to taste, 10 ms steps, until the arm hits on the
  beat you hear. Wired headphones and the speakers: nothing is hidden, the measured value is enough.
"""
import argparse
import sys
import time

import numpy as np

SR = 48000


def make_click(sr=SR, dur=0.012, f0=800.0, f1=9000.0):
    """A short windowed chirp: sharp, broadband autocorrelation (sub-ms timing), not harsh at low level."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    ph = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t ** 2 / dur)
    return (np.sin(ph) * np.hanning(n)).astype(np.float32)


def fit_timeline(pairs):
    """pairs: [(sample index of the first sample of a block, host time of it)] -> (a, b) with t = a + b * n."""
    p = np.array(pairs, float)
    b, a = np.polyfit(p[:, 0], p[:, 1], 1)
    return a, b


def detect_clicks(rec, click, sr, n_expected, min_gap=0.3):
    """Sample positions (float, sub-sample) of the clicks in `rec` (mono) by normalized cross-correlation."""
    n = len(rec)
    m = len(click)
    L = 1 << int(np.ceil(np.log2(n + m)))
    c = np.fft.irfft(np.fft.rfft(rec, L) * np.conj(np.fft.rfft(click, L)), L)[:n]  # c[k] = sum rec[k + j] click[j]
    # local energy of the recording under the template, for normalisation
    e = np.cumsum(np.r_[0.0, rec.astype(np.float64) ** 2])
    en = np.sqrt(np.maximum(e[m:] - e[:-m], 1e-12))
    ncc = np.zeros(n)
    ncc[:len(en)] = c[:len(en)] / (en * np.sqrt(np.sum(click.astype(np.float64) ** 2)))
    pos, thr = [], 0.0
    order = np.argsort(-np.abs(ncc))
    thr = 0.5 * np.abs(ncc[order[0]]) if len(order) else 0.0
    taken = np.zeros(n, bool)
    for i in order:
        if abs(ncc[i]) < thr or len(pos) >= n_expected:
            break
        if taken[max(0, i - int(min_gap * sr)):i + int(min_gap * sr)].any():
            continue
        taken[i] = True
        y0, y1, y2 = abs(ncc[i - 1]), abs(ncc[i]), abs(ncc[min(i + 1, n - 1)])
        d = y0 - 2 * y1 + y2
        pos.append(i + (0.5 * (y0 - y2) / d if d < 0 else 0.0))
    return np.sort(np.array(pos)), float(np.abs(ncc[order[0]])) if len(order) else 0.0


def analyze(rec, out_times, in_timeline, sr=SR, click=None, n_clicks=None):
    """rec: mono capture; out_times: host time each click reaches the DAC (s); in_timeline: (a, b) of the
    capture, t = a + b * n. Returns (L_bh per click in s, quality 0..1)."""
    click = make_click(sr) if click is None else click
    pos, q = detect_clicks(rec, click, sr, n_clicks or len(out_times))
    t_cap = in_timeline[0] + in_timeline[1] * pos
    # match each capture to the nearest scheduled click that is not later than it by more than 1 s
    res = []
    for tc in t_cap:
        j = int(np.argmin(np.abs(out_times - tc)))
        res.append(tc - out_times[j])
    return np.array(res), q


def report(L, q):
    if len(L) < 3:
        print(f"only {len(L)} click(s) found (peak correlation {q:.2f}): is the output routed to BlackHole, "
              "is the volume up, is another app using BlackHole?")
        return None
    med = float(np.median(L))
    mad = float(np.median(np.abs(L - med)))
    print(f"clicks found: {len(L)}   peak correlation {q:.2f}")
    print(f"L_bh (capture - DAC): median {med * 1000:+.2f} ms   MAD {mad * 1000:.2f} ms   "
          f"min {L.min() * 1000:+.2f}  max {L.max() * 1000:+.2f}")
    k = np.arange(len(L))
    drift = np.polyfit(k, L, 1)[0] * 1000 if len(L) > 3 else 0.0
    print(f"drift over the train: {drift:+.3f} ms per click (clock drift between the devices if not ~0)")
    if mad > 0.002:
        print("  spread above 2 ms: the capture is jittery (busy machine / buffer size), repeat when idle")
    print(f"\nsuggested for dance_live.py (wired headphones / speakers):   --output-delay {-med:+.3f}")
    print("Bluetooth (AirPods): the radio + codec delay (150-250 ms) is invisible here: use\n"
          f"   --headphones airpods --output-delay {-med + 0.18:+.3f}   then trim by ear in 10 ms steps")
    return med


def selftest():
    """Synthetic recordings with a known delay (no audio device): the analysis must recover it."""
    rng = np.random.default_rng(3)
    click = make_click()
    worst = 0.0
    for delay in (0.0373, 0.2131, -0.0120):
        n = int(14 * SR)
        rec = (0.02 * rng.standard_normal(n)).astype(np.float32)  # room / music floor (-34 dBFS)
        t0_host = 1000.123  # host time of the capture's first sample
        out_times = 1000.0 + 1.5 + 0.8 * np.arange(12) + 2.0e-4 * rng.standard_normal(12) * 0  # DAC times
        drift = 1.0 + 15e-6  # a 15 ppm clock difference between the devices
        for t in out_times:
            idx = (t + delay - t0_host) * SR * drift
            i = int(np.floor(idx))
            frac = idx - i
            # fractional delay by interpolation of the click
            ck = np.interp(np.arange(len(click)) - frac, np.arange(len(click)), click, left=0, right=0)
            rec[i:i + len(click)] += 0.3 * ck.astype(np.float32)
        # input timeline as PortAudio would give it: per-block host times with 0.2 ms jitter, 256-sample blocks
        blocks = [(k * 256, t0_host + k * 256 / (SR * drift) + 2e-4 * rng.standard_normal()) for k in range(n // 256)]
        tl = fit_timeline(blocks)
        L, q = analyze(rec, out_times, tl)
        err = float(np.median(L)) - delay
        worst = max(worst, abs(err))
        print(f"selftest delay {delay * 1000:+8.2f} ms -> measured {np.median(L) * 1000:+8.2f} ms "
              f"(error {err * 1000:+.3f} ms, {len(L)} clicks, corr {q:.2f})")
    print("selftest", "OK" if worst < 0.0005 else "FAILED", f"(worst error {worst * 1000:.3f} ms)")
    return worst < 0.0005


def find_device(sd, name, kind):
    devs = sd.query_devices()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    if name is None:
        return sd.default.device[0 if kind == "input" else 1]
    for i, d in enumerate(devs):
        if d[key] > 0 and name.lower() in d["name"].lower():
            return i
    raise SystemExit(f"no {kind} device matches {name!r}:\n" + "\n".join(
        f"  [{i}] {d['name']}" for i, d in enumerate(devs) if d[key] > 0))


def run(args):
    import sounddevice as sd
    out_dev = find_device(sd, args.output, "output")
    in_dev = find_device(sd, args.input, "input")
    print(f"output: [{out_dev}] {sd.query_devices(out_dev)['name']}   input: [{in_dev}] {sd.query_devices(in_dev)['name']}")
    click = make_click() * args.level
    n_total = int((args.lead_in + args.clicks * args.interval + 1.5) * SR)
    starts = (args.lead_in + args.interval * np.arange(args.clicks))
    start_idx = (starts * SR).astype(int)
    track = np.zeros(n_total + len(click), np.float32)
    for i in start_idx:
        track[i:i + len(click)] = click
    out_pairs, in_pairs = [], []
    rec = np.zeros(n_total, np.float32)
    state = {"n_out": 0, "n_in": 0}

    def out_cb(outdata, frames, ti, status):
        n = state["n_out"]
        chunk = track[n:n + frames]
        outdata[:, :] = 0
        outdata[:len(chunk), :] = chunk[:, None]
        out_pairs.append((n, ti.outputBufferDacTime))
        state["n_out"] = n + frames

    def in_cb(indata, frames, ti, status):
        n = state["n_in"]
        x = indata.mean(axis=1)
        k = min(frames, len(rec) - n)
        if k > 0:
            rec[n:n + k] = x[:k]
        in_pairs.append((n, ti.inputBufferAdcTime))
        state["n_in"] = n + frames

    oc = sd.query_devices(out_dev)["max_output_channels"]
    ins = sd.InputStream(device=in_dev, channels=min(2, sd.query_devices(in_dev)["max_input_channels"]),
                         samplerate=SR, blocksize=256, dtype="float32", callback=in_cb)
    outs = sd.OutputStream(device=out_dev, channels=min(2, oc), samplerate=SR, blocksize=256, dtype="float32",
                           callback=out_cb)
    print(f"PortAudio latency: output {outs.latency * 1000:.1f} ms, input {ins.latency * 1000:.1f} ms")
    print(f"playing {args.clicks} clicks (level {args.level:g}) ...")
    ins.start()
    outs.start()
    t_end = time.monotonic() + n_total / SR + 0.5
    while time.monotonic() < t_end:
        time.sleep(0.05)
    outs.stop()
    ins.stop()
    outs.close()
    ins.close()
    if args.save:
        np.savez(args.save, rec=rec, out_pairs=np.array(out_pairs), in_pairs=np.array(in_pairs),
                 start_idx=start_idx, level=args.level)
        print(f"saved {args.save}")
    return analyze_pairs(rec, out_pairs, in_pairs, start_idx, args.level)


def analyze_pairs(rec, out_pairs, in_pairs, start_idx, level=0.5):
    a_o, b_o = fit_timeline(out_pairs)
    in_tl = fit_timeline(in_pairs)
    out_times = a_o + b_o * np.asarray(start_idx, float)
    gap = abs((in_tl[0] + in_tl[1] * 0) - (a_o + b_o * 0))
    if gap > 5.0:
        print(f"WARNING: the input and output host clocks differ by {gap:.1f} s: PortAudio time stamps are not "
              "comparable on this setup, the result is meaningless")
    L, q = analyze(rec, out_times, in_tl, click=make_click() * level)
    return report(L, q)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", help="output device name substring (default: the system default output)")
    p.add_argument("--input", default="BlackHole", help="input device name substring (default BlackHole)")
    p.add_argument("--clicks", type=int, default=12)
    p.add_argument("--interval", type=float, default=0.8, help="s between clicks")
    p.add_argument("--lead-in", type=float, default=1.0, help="s of silence before the first click")
    p.add_argument("--level", type=float, default=0.3, help="click amplitude 0..1 (default 0.3, keep it moderate)")
    p.add_argument("--save", help="save the recording + time stamps (npz) for --analyze")
    p.add_argument("--analyze", help="analyse a saved recording instead of playing")
    p.add_argument("--selftest", action="store_true", help="check the analysis on synthetic recordings and exit")
    args = p.parse_args(argv)
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.analyze:
        z = np.load(args.analyze)
        analyze_pairs(z["rec"], z["out_pairs"], z["in_pairs"], z["start_idx"], float(z["level"]))
        return
    run(args)


if __name__ == "__main__":
    main()
