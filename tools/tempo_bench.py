#!/usr/bin/env python3
"""tempo_bench.py - offline beat references and a benchmark for the live tempo / beat clock of
dance_live.py (tools/dancelive/tempo.py).

    UV="uv run --no-project --python 3.12 --with librosa --with numpy"
    $UV tools/tempo_bench.py synth                      # synthetic test tracks (wav + exact beats)
    $UV tools/tempo_bench.py ref                        # offline references of ~/Desktop/Anime/music/*.mp3
    $UV tools/tempo_bench.py run --out DIR              # live tracker, hop by hop (as dance_live --fast)
    $UV tools/tempo_bench.py run --out DIR --tempo old_tempo.py     # ... with another tempo.py
    $UV tools/tempo_bench.py report DIR [DIR2]          # metrics table (DIR2: before/after side by side)

`report` also reads the logs of `dance_live.py --file X --fast --no-view --log DIR/<key>.npz`
(t, theta, bpm, conf at 50 Hz): name them after the reference keys (tempo_bench.py keys).

References (tools/.beat_reference/<key>.npz + summary.json): beats (s), local tempo curve, music
start, breaks / re-entries, clear-beat mask, the chosen metrical level and its alternatives, and a
confidence per song. Built with librosa (decode at 22050 Hz, hop 128): HPSS percussive and low-band
(kick / bass, < 150 Hz) onset envelopes; an autocorrelation tempogram ridge at the chosen level
(Viterbi, +-13 %) gives three tempo curves (constant, 15 s median, raw ridge); dynamic-programming
beat tracking (librosa, time-varying tempo) on the kick-weighted envelope with tightness 300..10000;
the rigid runs win (constant tempo, or the smooth curve when it aligns > 2 % better: a real tempo
change) unless they lose > 10 % of the onset alignment (rubato: the best aligned run). A flexible DP
gains a few % of alignment by sliding onto syncopated hits, which is exactly how the old librosa
reference (start_bpm, tightness 400) flipped to the off-beat for seconds. Beat times -20 ms
(REF_SHIFT, calibrated on the synthetic tracks). A beat is "clear" when it is salient, not silent,
and the percussive-only DP agrees (+-50 ms). Per song: the 16th-shift check (kick + broadband energy
at the 4 sixteenth shifts, window by window), agreement with the percussive-only DP and with the old
reference -> confidence high / medium / low. LEVELS documents the metrical level of each song.
Synthetic tracks (`synth`, deterministic) have exact references.

Metrics (per track):
  level %      share of time (every 0.5 s from 5 s on) the clock runs at the reference metrical level
               (slope of theta over 2 s within +-8 % of the local reference tempo; tempo4: +-4 %); no
               tempo counts as wrong
  lock s       time from the first clear reference beat to 4 consecutive beats with |phase error|
               < 0.12 beat at the right level
  phase ms     median / p90 of |phase error| at the reference beats after 20 s (clear beats only)
  off %        share of those beats where the clock is closer to an off-beat (|error| > 0.25 beat)
  recov s      time after each break re-entry until 4 beats are back within 0.12 beat (mean / max)
  conf r/AUC   correlation of the confidence with "right" (level ok and |error| < 0.12 beat) at the
               reference beats, and the probability that a right beat has a higher confidence than
               a wrong one (per track, and pooled over all tracks)
  p90 mod 1/2  the p90 counted modulo half a beat (the clock on a beat of the double level)
  legacy       the earlier benchmark's numbers (instantaneous slope +-2 % among seconds with a tempo;
               phase vs the plain librosa reference): in the --json output
  cpu ms       mean / p99 time of one TempoTracker._tempo_hop call (run only)
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
REF_DIR = TOOLS / ".beat_reference"
MUSIC = Path("~/Desktop/Anime/music").expanduser()
SR = 22050


def slug(name):
    """Reference key of a song file name: ASCII, no YouTube id."""
    base = Path(name).stem.split(" [")[0]
    if "VK Blanka" in base or base.startswith("ビッケブランカ"):
        return "VK_Blanka_live"
    s = "".join(c if c.isascii() and c.isalnum() else "_" for c in base)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")[:24] or "song"


def decode(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
                          "-ar", str(sr), "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def write_wav(path, y, sr=SR):
    y = np.clip(y, -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((y * 32767).astype("<i2").tobytes())


# ==================================================================== synthetic tracks
class Synth:
    """Tiny drum / bass / pad / piano synthesizer; events at exact sample times (the onset = the
    first sample of the event)."""

    def __init__(self, dur, sr=SR, seed=0):
        self.sr, self.y = sr, np.zeros(int(dur * sr) + sr, np.float64)
        self.rng = np.random.default_rng(seed)

    def _add(self, t0, x):
        i = int(round(t0 * self.sr))
        if i < 0 or i >= len(self.y):
            return
        n = min(len(x), len(self.y) - i)
        self.y[i:i + n] += x[:n]

    def _t(self, dur):
        return np.arange(int(dur * self.sr)) / self.sr

    def kick(self, t0, a=1.0):
        t = self._t(0.35)
        f = 48.0 + 110.0 * np.exp(-t / 0.03)
        x = np.sin(2 * np.pi * np.cumsum(f) / self.sr) * np.exp(-t / 0.12)
        x[:40] += np.linspace(1, 0, 40) * self.rng.uniform(-1, 1, 40) * 0.5  # beater click
        self._add(t0, a * x)

    def snare(self, t0, a=1.0):
        t = self._t(0.25)
        n = self.rng.normal(0, 1, len(t))
        n = np.diff(np.r_[0.0, n])  # brighter noise
        x = 0.5 * n * np.exp(-t / 0.07) + 0.6 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.04)
        self._add(t0, a * 0.8 * x)

    def clap(self, t0, a=1.0):
        t = self._t(0.2)
        n = np.diff(np.r_[0.0, self.rng.normal(0, 1, len(t))])
        env = np.exp(-t / 0.05) * (1 + 0.8 * np.sin(2 * np.pi * 90 * t) ** 8)
        self._add(t0, a * 0.7 * n * env)

    def hat(self, t0, a=1.0, open_=False):
        t = self._t(0.3 if open_ else 0.06)
        n = self.rng.normal(0, 1, len(t))
        n = np.diff(np.diff(np.r_[0.0, 0.0, n]))  # high-passed
        self._add(t0, a * 0.12 * n * np.exp(-t / (0.12 if open_ else 0.015)))

    def bass(self, t0, dur, f, a=1.0):
        t = self._t(dur)
        x = sum(np.sin(2 * np.pi * k * f * t) / k for k in range(1, 5))
        env = np.minimum(t / 0.004, 1.0) * np.exp(-t / 0.4) * np.minimum((dur - t) / 0.02, 1.0)
        self._add(t0, a * 0.35 * x * env)

    def pad(self, t0, dur, freqs, a=1.0):
        t = self._t(dur)
        x = sum(np.sin(2 * np.pi * f * t + p) + 0.5 * np.sin(2 * np.pi * f * 1.003 * t)
                for f, p in zip(freqs, self.rng.uniform(0, 6, len(freqs))))
        env = np.minimum(t / 0.6, 1.0) * np.minimum((dur - t) / 0.6, 1.0)
        self._add(t0, a * 0.05 * x * env)

    def piano(self, t0, f, a=1.0, dur=1.6):
        t = self._t(dur)
        x = sum(np.sin(2 * np.pi * k * f * t) * np.exp(-t * (1.2 + 0.9 * k)) / k for k in range(1, 7))
        x *= np.minimum(t / 0.003, 1.0) * np.minimum((dur - t) / 0.05, 1.0)
        self._add(t0, a * 0.3 * x)

    def stab(self, t0, freqs, a=1.0):
        """Short loud broadband chord hit (brass / synth stab) with a noise transient."""
        t = self._t(0.18)
        x = sum(np.sign(np.sin(2 * np.pi * f * t)) for f in freqs) / len(freqs)
        x = x * np.exp(-t / 0.06) + 0.6 * np.diff(np.r_[0.0, self.rng.normal(0, 1, len(t))]) * np.exp(-t / 0.02)
        self._add(t0, a * 0.5 * x)

    def voice(self, t0, dur, f, a=1.0):
        t = self._t(dur)
        ph = 2 * np.pi * np.cumsum(f * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * t))) / self.sr
        x = np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)
        env = np.minimum(t / 0.06, 1.0) * np.minimum((dur - t) / 0.08, 1.0)
        self._add(t0, a * 0.12 * x * env)

    def out(self):
        y = self.y / (np.abs(self.y).max() + 1e-9) * 0.7
        return y.astype(np.float32)


class Grid:
    """Beat position <-> time for a tempo curve bpm(t) (piecewise via a fine grid)."""

    def __init__(self, bpm_fn, dur, dt=0.001):
        self.tt = np.arange(0.0, dur + 1.0, dt)
        self.bb = np.r_[0.0, np.cumsum(bpm_fn(self.tt[:-1]) / 60.0 * dt)]

    def time(self, beat):
        return float(np.interp(beat, self.bb, self.tt))

    def beats_until(self, dur):
        return np.array([self.time(k) for k in range(int(self.bb[np.searchsorted(self.tt, dur)]) + 1)])


ROOTS = [55.0, 43.65, 49.0, 41.2]  # A1 F1 G1 E1
CHORDS = [[220, 261.6, 329.6], [174.6, 220, 261.6], [196, 246.9, 293.7], [164.8, 207.7, 246.9]]


def _groove(s, g, b0, b1, hats=8, kick_amp=1.0, snare_amp=0.8, bass_amp=1.0, fill=True):
    """Kick 1 & 3 (+ the 'and' of 3 in odd bars), snare 2 & 4, closed hats, bass on the beats."""
    for b in range(b0, b1):
        bar, k = divmod(b, 4)
        t = g.time(b)
        if k in (0, 2):
            s.kick(t, kick_amp)
        if k in (1, 3):
            s.snare(t, snare_amp)
        if fill and k == 2 and bar % 2:
            s.kick(g.time(b + 0.5), 0.6 * kick_amp)
        for j in range(hats // 4):
            s.hat(g.time(b + j * 4.0 / hats), 0.9 if j == 0 else 0.6)
        s.bass(t, 0.9 * (g.time(b + 1) - t), ROOTS[bar % 4] * (2 if k == 3 else 1), bass_amp)
        if k == 0:
            s.pad(t, g.time(b + 4) - t, CHORDS[bar % 4])


def synth_tracks():
    """name -> (audio, info). info: beats, bpm curve (t, bpm), breaks [(start, end)], level."""
    out = {}
    for bpm in (90, 120, 140):
        dur = 75.0
        g = Grid(lambda t, b=bpm: np.full_like(t, b, dtype=float), dur)
        s = Synth(dur, seed=bpm)
        nb = int(dur * bpm / 60) - 2
        _groove(s, g, 4, nb, hats=16 if bpm <= 100 else 8)
        for b in range(0, 4):  # one bar of count-in hats
            s.hat(g.time(b), 0.8)
        bt = g.beats_until(dur)[:nb]
        out[f"synth_steady{bpm}"] = (s.out(), dict(beats=bt, start=bt[0], breaks=[], level=bpm))
    # tempo change 100 -> 128 at bar 16 of the first part
    t_sw = 16 * 4 * 60 / 100.0
    dur = t_sw + 40.0
    g = Grid(lambda t: np.where(t < t_sw, 100.0, 128.0), dur)
    s = Synth(dur, seed=7)
    nb = int(g.bb[np.searchsorted(g.tt, dur)]) - 2
    _groove(s, g, 0, nb)
    bt = g.beats_until(dur)[:nb]
    out["synth_change100_128"] = (s.out(), dict(beats=bt, start=bt[0], breaks=[], level=None, change=t_sw))
    # rubato piano ballad: 76 bpm +-8 % (two slow sines), chord on the beat, soft 8th arpeggio, melody
    dur = 90.0
    rng = np.random.default_rng(3)
    f_rub = lambda t: 76.0 * (1 + 0.055 * np.sin(2 * np.pi * t / 23.0 + 0.4) + 0.03 * np.sin(2 * np.pi * t / 9.0 + 1.3))
    g = Grid(f_rub, dur)
    s = Synth(dur, seed=11)
    nb = int(g.bb[np.searchsorted(g.tt, dur)]) - 2
    for b in range(nb):
        bar, k = divmod(b, 4)
        ch = CHORDS[bar % 4]
        t = g.time(b)
        s.piano(t, ROOTS[bar % 4] * 2, 1.0 if k == 0 else 0.7)
        for f in ch:
            s.piano(t, f, 0.45 if k == 0 else 0.3)
        s.piano(g.time(b + 0.5), ch[(b % 3)] * 2, 0.25)  # soft 8th arpeggio
        if rng.random() < 0.6:  # melody: on the beat, the and, or a tied anticipation
            pos = b + rng.choice([0.0, 0.5, 0.75, 1.5 / 2])
            s.voice(g.time(pos), 0.6 * (g.time(b + 1) - t), ch[-1] * 2 * rng.choice([1.0, 1.122, 1.335]), 1.0)
    # reference at the 8th-note level (~152), the level the live tracker aims at (the fastest clear pulse
    # up to 165 bpm) and the one of the real ballad (If I Am With You, 162 = its 8th notes)
    bt = np.array([g.time(0.5 * k) for k in range(2 * nb)])
    out["synth_rubato76"] = (s.out(), dict(beats=bt, start=bt[0], breaks=[], level=None, rubato=True))
    # syncopated: the loudest onsets (stabs + claps) on 16th off-beats, kick/snare softer on the beats
    bpm, dur = 118.0, 75.0
    g = Grid(lambda t: np.full_like(t, bpm), dur)
    s = Synth(dur, seed=5)
    nb = int(dur * bpm / 60) - 2
    _groove(s, g, 0, nb, hats=8, kick_amp=0.55, snare_amp=0.45, fill=False)
    for bar in range(nb // 4):
        for p in (0.75, 1.75, 2.25, 3.5 + 0.25):  # 'a' of 1, 'a' of 2, 'e' of 3, 'a' of 4
            s.stab(g.time(4 * bar + p), [c * 2 for c in CHORDS[bar % 4]], 1.0)
            s.clap(g.time(4 * bar + p), 0.8)
    bt = g.beats_until(dur)[:nb]
    out["synth_syncopated118"] = (s.out(), dict(beats=bt, start=bt[0], breaks=[], level=bpm))
    # breaks: 16 s groove / 4 s break (pad + a free melody, no drums), tempo kept through the breaks
    bpm, dur = 124.0, 104.0
    g = Grid(lambda t: np.full_like(t, bpm), dur)
    s = Synth(dur, seed=9)
    nb = int(dur * bpm / 60) - 2
    bt = g.beats_until(dur)[:nb]
    breaks = []
    rng = np.random.default_rng(4)
    for b in range(nb):
        t = bt[b]
        in_break = (t % 20.0) >= 16.0
        if in_break:
            if not breaks or breaks[-1][1] < t - 1.0:
                b_end = b
                while b_end < nb and (bt[b_end] % 20.0) >= 16.0:
                    b_end += 1
                breaks.append([float(t), float(bt[min(b_end, nb - 1)])])
            if b % 4 == 0:
                s.pad(t, g.time(b + 4) - t, CHORDS[(b // 4) % 4], 1.5)
            if rng.random() < 0.7:  # rubato-ish melody: notes at triplet / off-beat positions
                p = b + rng.choice([0.33, 0.5, 0.67, 0.8])
                s.voice(g.time(p), 0.3, CHORDS[(b // 4) % 4][-1] * 2 * rng.choice([1.0, 1.122, 1.26]), 1.4)
        else:
            _groove(s, g, b, b + 1)
    out["synth_breaks124"] = (s.out(), dict(beats=bt, start=bt[0], breaks=breaks, level=bpm))
    # octave ambiguity at 167 bpm (just above the tracker's 165 cut-off: it is held at 167 or 83.5):
    # 12 s full groove (kick-snare backbeat on the 167 grid, 8th hats) / 12 s sparse half-time feel
    # (kick 1, snare 3, nothing between), repeated; the beat grid never changes. The tracker before
    # the octave hysteresis flipped 83 <-> 167 at most section changes (the live report)
    bpm, dur = 167.0, 110.0
    g = Grid(lambda t: np.full_like(t, bpm), dur)
    s = Synth(dur, seed=13)
    nb = int(dur * bpm / 60) - 2
    bt = g.beats_until(dur)[:nb]
    for b in range(nb):
        bar, k = divmod(b, 4)
        t = bt[b]
        if int(t // 12.0) % 2 == 0:  # full groove
            (s.kick if k in (0, 2) else s.snare)(t, 1.0 if k in (0, 2) else 0.8)
            s.hat(t, 0.8)
            s.hat(g.time(b + 0.5), 0.5)
        else:  # sparse half time
            if k == 0:
                s.kick(t, 1.0)
            if k == 2:
                s.snare(t, 0.8)
        if k == 0:
            s.bass(t, 0.9 * (g.time(b + 2) - t), ROOTS[(bar // 2) % 4])
    out["synth_octave167"] = (s.out(), dict(beats=bt, start=bt[0], breaks=[], level=bpm))
    return out


def cmd_synth(args):
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, (y, info) in synth_tracks().items():
        write_wav(d / f"{name}.wav", y)
        beats = np.asarray(info["beats"], float)
        ibi = np.diff(beats)
        tb = 0.5 * (beats[1:] + beats[:-1])
        save_ref(d / f"{name}.npz", dict(
            key=name, audio=str(d / f"{name}.wav"), beats=beats, tempo_t=tb, tempo_bpm=60.0 / ibi,
            start=float(info["start"]), breaks=np.array(info["breaks"], float).reshape(-1, 2),
            clear=np.ones(len(beats), bool), level=float(info["level"] or np.median(60.0 / ibi)),
            alt_levels=np.array([83.5] if name == "synth_octave167" else []), confidence="exact", synthetic=True))
        print(f"{name}: {len(y) / SR:.0f} s, {len(beats)} beats, {len(info['breaks'])} breaks")


def save_ref(path, d):
    np.savez(path, **{k: (np.asarray(v) if not isinstance(v, (str, bool, float, int)) else np.array(v))
                      for k, v in d.items()})


def load_ref(path):
    z = np.load(path, allow_pickle=False)
    r = {k: z[k] for k in z.files}
    for k in ("key", "audio", "confidence"):
        if k in r:
            r[k] = str(r[k])
    return r


# ==================================================================== offline references
# Metrical level of each benchmark song: the pulse a listener taps, at most 165 bpm like the live
# tracker (faster pulses are tracked at half: dance_live dances half time above --max-dance-bpm
# anyway). alt = other defensible levels (a clock there is a level error, but not a musical
# nonsense). The reasons come from the beat-synchronous onset profiles (16th / triplet grid fit,
# beat accent, kick / snare alternation) printed by `ref --verbose`.
LEVELS = {
    "1991": (124.6, [62.3], "shuffle (triplet subdivision); 93.1 = the same tatum grouped by 4 has a weak beat accent"),
    "AIZO": (94.9, [189.8], "dense 16ths, the 8th level (190, above the 165 cap) is nearly as strong: beat 1.8 vs 'and' 1.6"),
    "Defeat_here": (148.0, [74.0], "kick / snare alternate at 148 (backbeat); 197.5 = dotted-8th (3-3-2) syncopation"),
    "Delirious": (143.8, [71.9], "snare backbeat at 143.8; 191.8 = dotted 8ths"),
    "If_I_Am_With_You": (162.0, [81.0], "rubato ballad: the clearest pulse is the 8th-note piano (161.5); 81 = quarter notes"),
    "Malevolent_Shrine": (116.0, [58.0], "16th grid at 116; 154.8 / 77.3 come from 3-3-2 dotted-8th syncopations"),
    "My_Hero_Academia_OST_You": (87.6, [175.2], "backbeat at 175 (above the cap): half time"),
    "Ripple": (139.0, [69.5], "triplet subdivision; 103 (the same tatum grouped by 4) has a much weaker beat accent"),
    "TACTIC": (87.0, [174.0], "16ths; the 8th level (174, above the cap) is almost as strong"),
    "Thunderclap": (146.0, [73.0], "weak 16th grid at 146; 195 = dotted 8ths"),
    "Yuru_Camp_OST_02_Campsit": (119.6, [59.8], "clear beat"),
    "natori_Absolute_Zero": (110.7, [221.4], "16th grid at 110.7; 147.6 = dotted 8ths"),
    "VK_Blanka_live": (125.0, [62.5], "live recording (tempo drifts a little), kick on the beat, open hats off-beat"),
}
REF_HOP = 128
REF_SHIFT = -0.020  # s added to the DP beat times: librosa's beats come ~20 ms after the true onsets
# (synthetic drums: +17..+24 ms, MAD 1.5 ms, measured with build_reference on the `synth` tracks)


def _nz(x):
    return x / (np.std(x) + 1e-9)


def ref_features(y, sr=SR, hop=REF_HOP):
    """Onset envelopes (frames at hop / sr) of the HPSS percussive part: all bands, kick / bass
    (< 150 Hz), mid (150 Hz - 4 kHz), high (> 4 kHz); plus the full-mix envelope (librosa default)."""
    import librosa
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    _, P = librosa.decompose.hpss(D, kernel_size=(17, 31))
    fb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=128, fmin=20.0, fmax=sr / 2)
    mf = librosa.mel_frequencies(128, fmin=20.0, fmax=sr / 2)
    Lp = librosa.power_to_db(fb @ (P ** 2))
    oe = lambda sel: librosa.onset.onset_strength(S=Lp[sel], sr=sr, hop_length=hop)  # noqa: E731
    env = dict(perc=oe(slice(None)), plow=oe(mf < 150), pmid=oe((mf >= 150) & (mf < 4000)), phigh=oe(mf >= 4000),
               full=librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop),
               rms=librosa.feature.rms(S=D, frame_length=2048, hop_length=hop)[0])
    n = min(len(v) for v in env.values())
    return {k: v[:n].astype(np.float64) for k, v in env.items()}, sr / hop


def tempo_ridge(x, fps, level, band=0.13, dec=2, win=8.0, jump=4.0):
    """Local tempo (bpm per frame of x) near `level`: autocorrelation tempogram of x (decimated by
    `dec`), the best path of lags within +-band of the level (Viterbi, a cost on log-tempo jumps),
    parabolic sub-lag refinement, 3 s median smoothing."""
    import librosa
    from scipy.ndimage import median_filter
    n = len(x) // dec * dec
    xd = x[:n].reshape(-1, dec).mean(1)
    fd = fps / dec
    tg = librosa.feature.tempogram(onset_envelope=xd, sr=SR, hop_length=REF_HOP * dec, win_length=int(win * fd))
    lags = np.arange(tg.shape[0])
    with np.errstate(divide="ignore"):
        bpm = 60.0 * fd / lags
    sel = np.where((bpm >= level / (1 + band)) & (bpm <= level * (1 + band)))[0]
    E = tg[sel]  # (states, frames)
    E = (E - E.mean(0)) / (E.std(0) + 1e-9)
    lb = np.log(bpm[sel])
    trans = -jump * np.abs(lb[:, None] - lb[None, :]) / 0.01  # per 1 % jump
    T = E.shape[1]
    score = E[:, 0].copy()
    back = np.zeros((len(sel), T), np.int32)
    for t in range(1, T):
        cand = score[None, :] + trans
        back[:, t] = np.argmax(cand, axis=1)
        score = cand[np.arange(len(sel)), back[:, t]] + E[:, t]
    path = np.zeros(T, np.int64)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    li = sel[path].astype(float)
    for t in range(T):  # sub-lag
        k = int(li[t])
        if 1 <= k < tg.shape[0] - 1:
            y0, y1, y2 = tg[k - 1, t], tg[k, t], tg[k + 1, t]
            d = y0 - 2 * y1 + y2
            if d < 0:
                li[t] = k + 0.5 * (y0 - y2) / d
    curve = median_filter(60.0 * fd / li, size=max(3, int(3.0 * fd) | 1), mode="nearest")
    return np.interp(np.arange(len(x)) / fps, np.arange(T) / fd, curve)


def _at(x, fps, times, w=0.035):
    """max of x within +-w s of each time"""
    k = int(round(w * fps))
    idx = np.round(np.asarray(times) * fps).astype(int)
    return np.array([x[max(i - k, 0):i + k + 1].max() if 0 <= i < len(x) else 0.0 for i in idx])


def beat_profile(beats, x, fps, P=12):
    """Mean of x at P positions per beat (mean within +-1/(2P) beat)."""
    out = np.zeros(P)
    for b0, b1 in zip(beats[:-1], beats[1:]):
        L = b1 - b0
        for k in range(P):
            c = b0 + k * L / P
            a, b = int((c - L / (2 * P)) * fps), int((c + L / (2 * P)) * fps) + 1
            out[k] += x[max(a, 0):b].mean() if b > a else 0.0
    return out / max(len(beats) - 1, 1)


def fmeasure(a, b, tol=0.05):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    j = np.searchsorted(b, a)
    d = np.minimum(np.abs(a - b[np.clip(j, 0, len(b) - 1)]), np.abs(a - b[np.clip(j - 1, 0, len(b) - 1)]))
    hit = float((d <= tol).sum())
    return 2 * hit / (len(a) + len(b))


def auto_level(env, fps):
    """Level for a song not in LEVELS: the strongest autocorrelation peak of the kick-weighted
    envelope, moved by octaves into 70-165 bpm."""
    x = _nz(env["perc"]) + _nz(env["plow"])
    x = x[: len(x) // 4 * 4].reshape(-1, 4).mean(1) - x.mean()
    f4 = fps / 4
    n = len(x)
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * n)) ** 2)[:n]
    ac /= ac[0]
    lo, hi = int(60 * f4 / 200), int(60 * f4 / 50)
    L = lo + int(np.argmax(ac[lo:hi]))
    b = 60 * f4 / L
    while b > 165:
        b /= 2
    while b < 70:
        b *= 2
    return b


def build_reference(y, key, level=None, alt=(), note="", verbose=False):
    import librosa
    env, fps = ref_features(y)
    auto = auto_level(env, fps)
    if level is None:
        level, note = auto, "automatic (strongest periodicity folded into 70-165 bpm)"
    x = _nz(env["perc"]) + _nz(env["plow"])  # kick-weighted: the DP prefers beats with a kick / bass onset
    from scipy.ndimage import median_filter
    ridge = tempo_ridge(x, fps, level, band=0.13)
    n = len(ridge)
    dur = n / fps
    # tempo curve candidates: constant (produced music), slowly varying (15 s median: a tempo change,
    # a live band), the local ridge (rubato). DP beat tracking with each and several tightness values.
    # A flexible DP always gains a few % of onset alignment by sliding onto syncopated hits (it then
    # flips to the off-beat or a 16th in dense passages), so the rigid runs (tightness 10000) win:
    # constant tempo, or the smooth curve when it aligns > 2 % better (a real tempo change), unless
    # they lose more than 10 % of the alignment (rubato: then the best aligned run)
    variants = [("constant", np.full(n, float(np.median(ridge[int(0.1 * n):int(0.9 * n)])))),
                ("smooth", median_filter(ridge, size=int(15 * fps) | 1, mode="nearest")), ("ridge", ridge)]
    runs = {}
    for vi, (_, cv) in enumerate(variants):
        for tt in (10000, 3000, 1000, 300):
            _, b = librosa.beat.beat_track(onset_envelope=x, sr=SR, hop_length=REF_HOP, bpm=cv, tightness=tt,
                                           trim=False, units="time")
            runs[(vi, tt)] = np.asarray(b) + REF_SHIFT
    align = {k: float(np.mean(_at(x, fps, b))) for k, b in runs.items()}
    best = max(align.values())
    rigid = (1, 10000) if align[(1, 10000)] > 1.02 * align[(0, 10000)] else (0, 10000)  # a real tempo change
    vi, tt = rigid if align[rigid] >= 0.90 * best else max(runs, key=lambda k: align[k])
    variant, curve = variants[vi]
    beats = runs[(vi, tt)]
    # cross-checks: percussive-only envelope, and the old reference method (plain librosa, start_bpm, 400)
    _, b_perc = librosa.beat.beat_track(onset_envelope=env["perc"], sr=SR, hop_length=REF_HOP, bpm=curve,
                                        tightness=tt, trim=False, units="time")
    _, b_old = librosa.beat.beat_track(onset_envelope=env["full"], sr=SR, hop_length=REF_HOP, start_bpm=level,
                                       tightness=400, trim=False, units="time")
    b_perc, b_old = np.asarray(b_perc) + REF_SHIFT, np.asarray(b_old)  # b_old: exactly the old method (legacy)
    j = np.clip(np.searchsorted(b_perc, beats), 1, len(b_perc) - 1)
    agree_b = np.minimum(np.abs(beats - b_perc[j - 1]), np.abs(beats - b_perc[j])) < 0.05
    # beat clarity: kick-weighted onset at the beat vs the mean of the envelope around it (+-4 beats)
    s = _at(x, fps, beats)
    mloc = np.array([x[max(int((t - 2.0) * fps), 0):int((t + 2.0) * fps)].mean() for t in beats])
    rms_db = 20 * np.log10(_at(env["rms"], fps, beats, 0.2) + 1e-9)
    sal = np.convolve(s / (mloc + 1e-9), np.ones(8) / 8, mode="same")
    # clear beat: salient, not silent, and both DPs (kick-weighted / percussive only) put a beat there
    clear = (sal > 1.6) & (rms_db > rms_db.max() - 35) & agree_b
    # phase check every 8 beats: kick + broadband at the 4 sixteenth shifts of the grid
    win_flags = []
    for i in range(0, len(beats) - 9, 4):
        bb = beats[i:i + 9]
        ibi = np.diff(bb).mean()
        sc = []
        for sh in (0.0, 0.25, 0.5, 0.75):
            tq = bb + sh * ibi
            sc.append(np.mean(_at(_nz(env["plow"]), fps, tq)) + np.mean(_at(_nz(env["perc"]), fps, tq)))
        sc = np.array(sc)
        win_flags.append((float(bb[0]), int(np.argmax(sc)), float(sc[0] / (sc[1:].max() + 1e-9))))
    clear_w = [f for f in win_flags if clear[np.searchsorted(beats, f[0])]]
    suspect = float(np.mean([f[1] != 0 for f in clear_w])) if clear_w else 1.0
    margin = float(np.median([f[2] for f in clear_w])) if clear_w else 0.0
    # music start, breaks (the kick-weighted envelope drops under 45 % of its median for >= 2 s)
    i0 = next((i for i in range(len(clear) - 4) if clear[i:i + 4].all()), 0)
    start = float(beats[i0])
    k = int(2.0 * fps)
    acts = []
    for name in ("plow", "perc"):
        a_ = np.convolve(env[name], np.ones(k) / k, mode="same")
        acts.append(a_ / (np.median(a_[int(start * fps):int(beats[-1] * fps)]) + 1e-9))
    A = np.minimum(*acts)  # drums / bass gone: both the kick band and the percussive part drop
    breaks, j = [], int(start * fps)
    while j < int((dur - 3.0) * fps):
        if A[j] < 0.3:
            e = j
            while e < len(A) and A[e] < 0.5:
                e += 1
            if (e - j) / fps >= 2.0 and e / fps < dur - 3.0:
                breaks.append([j / fps, e / fps])
            j = e + 1
        else:
            j += 1
    ibi = np.diff(beats)
    tb = 0.5 * (beats[1:] + beats[:-1])
    loc = np.array([60.0 / np.median(ibi[max(0, i - 4):i + 5]) for i in range(len(ibi))])
    prof = beat_profile(beats[clear[:len(beats)]] if clear.sum() > 16 else beats, env["perc"], fps)
    prof /= prof.mean()
    agree_perc = fmeasure(beats, b_perc)
    agree_old = fmeasure(beats[beats > 20], b_old[b_old > 20])
    conf = ("high" if agree_perc > 0.9 and suspect < 0.15 and margin > 1.15 else
            "low" if agree_perc < 0.75 or suspect > 0.3 or margin < 1.05 else "medium")
    info = dict(key=key, level=float(level), alt_levels=[float(a) for a in alt], note=note, auto_level=float(auto),
                tempo_curve=variant, tightness=int(tt),
                align={f"{variants[k[0]][0]}/{k[1]}": round(v, 3) for k, v in align.items()},
                median_bpm=float(np.median(loc)), bpm_range=[float(np.percentile(loc, 5)), float(np.percentile(loc, 95))],
                n_beats=int(len(beats)), clear_frac=float(clear.mean()), start=start, breaks=breaks,
                phase_suspect_frac=suspect, phase_margin=margin, agree_perc_dp=agree_perc, agree_old_ref=agree_old,
                profile_12=[round(float(v), 2) for v in prof], confidence=conf)
    if verbose:
        print(json.dumps({k: v for k, v in info.items() if k != "alt_levels"}, ensure_ascii=False))
        for f in win_flags:
            if f[1] != 0 and clear[np.searchsorted(beats, f[0])]:
                print(f"   phase suspect at {f[0]:6.1f}s: best shift {f[1] / 4:.2f} beat (beat / best other {f[2]:.2f})")
    ref = dict(key=key, beats=beats, tempo_t=tb, tempo_bpm=loc, start=start, breaks=np.array(breaks, float).reshape(-1, 2),
               clear=clear, level=float(level), alt_levels=np.array(alt, float), confidence=conf, synthetic=False,
               curve_t=np.arange(0, len(curve), 43) / fps, curve_bpm=curve[::43], beats_perc_dp=b_perc, beats_old=b_old)
    return ref, info


def _ref_one(job):
    path, out_dir, verbose = job
    key = slug(Path(path).name)
    lv = LEVELS.get(key)
    y = decode(path)
    t0 = time.time()
    ref, info = build_reference(y, key, *(lv if lv else (None, (), "")), verbose=verbose)
    ref["audio"] = str(path)
    info["file"] = Path(path).name
    save_ref(Path(out_dir) / f"{key}.npz", ref)
    info["build_s"] = round(time.time() - t0, 1)
    return key, info


def cmd_ref(args):
    from concurrent.futures import ProcessPoolExecutor
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = args.files or sorted(glob.glob(str(MUSIC / "*.mp3")))
    summ_path = out / "summary.json"
    summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    jobs = [(f, str(out), args.verbose) for f in files]
    with ProcessPoolExecutor(max(1, args.jobs)) as ex:
        for key, info in ex.map(_ref_one, jobs):
            summ[key] = info
            print(f"{key:26s} level {info['level']:6.1f} (auto {info['auto_level']:6.1f}) median {info['median_bpm']:6.1f} "
                  f"[{info['bpm_range'][0]:.1f}-{info['bpm_range'][1]:.1f}] tightness {info['tightness']:5d} "
                  f"clear {info['clear_frac']:.2f} suspect {info['phase_suspect_frac']:.2f} margin {info['phase_margin']:.2f} "
                  f"agree perc-DP {info['agree_perc_dp']:.2f} old-ref {info['agree_old_ref']:.2f} breaks {len(info['breaks'])} "
                  f"-> {info['confidence']}", flush=True)
    summ_path.write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(f"summary: {summ_path}")


# ==================================================================== live tracker runs
def load_analyzer(tempo_path=None):
    """LiveAnalyzer of tools/dancelive; with tempo_path, the same package with another tempo.py (a
    throwaway package in a temp dir that links every other module of dancelive)."""
    if tempo_path is None:
        sys.path.insert(0, str(TOOLS))
        from dancelive.analysis import LiveAnalyzer
        return LiveAnalyzer
    import atexit
    import importlib
    d = Path(tempfile.mkdtemp(prefix="tempo_bench_"))
    atexit.register(shutil.rmtree, d, True)
    name = "tbpkg_" + d.name.rsplit("_", 1)[-1]
    pkg = d / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    src = Path(tempo_path).resolve()
    for f in (TOOLS / "dancelive").glob("*.py"):
        if f.name != "__init__.py" and not (src.parent / f.name).exists():
            (pkg / f.name).symlink_to(f)
    for f in src.parent.glob("*.py"):  # tempo.py and its helper modules (onsets.py, ...)
        if f.name != "__init__.py" and (f.name == "tempo.py" or f.name.startswith("onsets")):
            shutil.copy(f, pkg / f.name)
    if src.name != "tempo.py":
        shutil.copy(src, pkg / "tempo.py")
    for f in (TOOLS / "dancelive").glob("*.py"):
        if f.name != "__init__.py" and not (pkg / f.name).exists():
            (pkg / f.name).symlink_to(f)
    sys.path.insert(0, str(d))
    return importlib.import_module(name + ".analysis").LiveAnalyzer


def run_track(y, LiveAnalyzer, sr=SR, max_dance_bpm=124.0, params=None):
    """The analysis as dance_live --fast runs it (one process() per hop); the clock read at the
    stream time after each hop. params: attributes set on the analyzer before the first hop
    (parameter sweeps). Returns the log dict."""
    an = LiveAnalyzer(sr, max_dance_bpm=max_dance_bpm)
    for k, v in (params or {}).items():
        setattr(an, k, v)
    inner = an._tempo_hop
    cost = []

    def timed(*a, **k):
        t0 = time.perf_counter()
        r = inner(*a, **k)
        cost.append(time.perf_counter() - t0)
        return r

    an._tempo_hop = timed
    n = len(y) // an.hop
    L = {k: np.zeros(n) for k in ("t", "theta", "bpm", "conf", "locked", "div")}
    for k in range(n):
        an.process(y[k * an.hop:(k + 1) * an.hop])
        ts = (k + 1) * an.hop / sr
        L["t"][k] = ts
        L["theta"][k] = an.theta_ref + (ts - an.t_ref) * an.rate
        L["bpm"][k] = an.music_bpm or 0.0
        L["conf"][k] = an.conf
        L["locked"][k] = an.locked
        L["div"][k] = an.div
    L["cpu_ms"] = np.array(cost) * 1e3
    return L


def tracks(ref_dir, synth_dir, only=None):
    """key -> (audio path, reference path) for every song reference and synthetic track."""
    out = {}
    for d in (ref_dir, synth_dir):
        if d is None or not Path(d).exists():
            continue
        for p in sorted(Path(d).glob("*.npz")):
            r = load_ref(p)
            f = r.get("audio")
            if f and Path(f).exists() and (not only or any(o in r["key"] for o in only)):
                out[r["key"]] = (f, str(p))
    return out


def _run_one(job):
    key, audio, out, tempo_path = job
    LA = load_analyzer(tempo_path)
    y = decode(audio)
    t0 = time.time()
    L = run_track(y, LA)
    np.savez_compressed(Path(out) / f"{key}.npz", **L)
    return key, time.time() - t0, float(L["cpu_ms"].mean()), float(np.percentile(L["cpu_ms"], 99))


def cmd_run(args):
    from concurrent.futures import ProcessPoolExecutor
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tr = tracks(args.refs, args.synth, args.only)
    jobs = [(k, a, args.out, args.tempo) for k, (a, _) in tr.items()]
    with ProcessPoolExecutor(max(1, args.jobs)) as ex:
        for key, el, cm, c99 in ex.map(_run_one, jobs):
            print(f"{key:26s} {el:5.1f} s   tempo hop {cm:.3f} ms mean, {c99:.3f} ms p99", flush=True)


# ==================================================================== metrics
def evaluate(L, ref):
    t, th = np.asarray(L["t"], float), np.asarray(L["theta"], float)
    has = np.asarray(L["bpm"], float) > 0
    conf = np.asarray(L["conf"], float)
    beats = np.asarray(ref["beats"], float)
    beats = beats[beats < t[-1] - 1.0]
    clear = np.asarray(ref["clear"], bool)[:len(beats)]
    start = float(ref["start"])
    th_at = lambda x: np.interp(x, t, th)  # noqa: E731
    r = {}
    # tempo level, every 0.5 s of music from 5 s on: slope of the clock over 2 s vs the local reference
    ts = np.arange(max(5.0, start), min(t[-1], beats[-1]) - 1.0, 0.5)
    if clear.any():  # only where the reference has a clear beat (+-1 s)
        cb = beats[clear]
        k = np.clip(np.searchsorted(cb, ts), 1, len(cb) - 1)
        ts = ts[np.minimum(np.abs(ts - cb[k - 1]), np.abs(ts - cb[k])) < 1.0]
    slope = (th_at(ts + 1.0) - th_at(ts - 1.0)) / 2.0 * 60.0
    rb = np.interp(ts, ref["tempo_t"], ref["tempo_bpm"])
    hz = np.interp(ts, t, has.astype(float)) > 0.5
    ratio = np.where(hz, slope / rb, 0.0)
    ok_t = hz & (np.abs(ratio - 1) < 0.08)  # the metrical level (the next levels are >= 25 % away)
    r["level"] = 100 * ok_t.mean()
    r["tempo4"] = 100 * np.mean(hz & (np.abs(ratio - 1) < 0.04))  # tempo within 4 %
    r["x2"] = 100 * np.mean(hz & (np.abs(ratio / 2 - 1) < 0.08))
    r["x05"] = 100 * np.mean(hz & (np.abs(ratio * 2 - 1) < 0.08))
    r["none"] = 100 * np.mean(~hz)
    # octave / level flips of the music clock: tempo (4 s windows, every 1 s) jumping > 15 %
    tf = np.arange(max(start, 2.0) + 2.0, t[-1] - 2.0, 1.0)
    sf = (th_at(tf + 2.0) - th_at(tf - 2.0)) / 4.0 * 60.0
    hf = np.interp(tf, t, has.astype(float)) > 0.5
    jump = hf[1:] & hf[:-1] & (np.abs(sf[1:] / np.maximum(sf[:-1], 1e-6) - 1.0) > 0.15)
    r["flips"] = int(np.sum(jump & ~np.r_[False, jump[:-1]]))  # count each jump once
    # phase at the reference beats
    b = th_at(beats)
    e = b - np.round(b)  # clock beats
    ibi = np.gradient(beats)
    lvl_b = np.interp(beats, ts, ok_t.astype(float), left=0, right=0) > 0.5 if len(ts) else np.zeros(len(beats), bool)
    good = lvl_b & (np.abs(e) < 0.12)
    sel = (beats > 20.0) & clear
    err = np.abs(e) * ibi * 1000
    r["med"] = float(np.median(err[sel])) if sel.any() else np.nan
    r["p90"] = float(np.percentile(err[sel], 90)) if sel.any() else np.nan
    r["off"] = 100 * float(np.mean(np.abs(e[sel]) > 0.25)) if sel.any() else np.nan
    e2 = 2 * b - np.round(2 * b)  # modulo half a beat: on a beat of the double level
    r["p90_2x"] = float(np.percentile(np.abs(e2[sel]) * ibi[sel] * 500, 90)) if sel.any() else np.nan
    # lock time
    i0 = int(np.searchsorted(beats, start))
    lock = next((beats[i] - start for i in range(i0, len(beats) - 4) if good[i:i + 4].all()), np.nan)
    r["lock"] = float(lock)
    # recovery after each break
    rec = []
    for s0, s1 in np.asarray(ref["breaks"], float).reshape(-1, 2):
        j0 = int(np.searchsorted(beats, s1 - 0.02))
        jj = next((j for j in range(j0, len(beats) - 4) if good[j:j + 4].all()), None)
        rec.append(beats[jj] - s1 if jj is not None else t[-1] - s1)
    r["recov"] = float(np.mean(rec)) if rec else np.nan
    r["recov_max"] = float(np.max(rec)) if rec else np.nan
    r["n_breaks"] = len(rec)
    # confidence vs being right, at the beats after the music start (inside the evaluated span)
    cs = (beats > start + 5.0) & (beats >= ts[0]) & (beats <= ts[-1]) if len(ts) else beats < 0
    c = np.interp(beats[cs], t, conf)
    right = good[cs]
    if right.any() and (~right).any() and c.std() > 0:
        r["conf_r"] = float(np.corrcoef(c, right.astype(float))[0, 1])
        from scipy.stats import mannwhitneyu
        r["auc"] = float(mannwhitneyu(c[right], c[~right]).statistic / (right.sum() * (~right).sum()))
    else:
        r["conf_r"], r["auc"] = np.nan, np.nan
    r["_conf"], r["_right"] = c, right
    r["conf_right"] = float(c[right].mean()) if right.any() else np.nan
    r["conf_wrong"] = float(c[~right].mean()) if (~right).any() else np.nan
    r["right"] = 100 * float(right.mean()) if cs.any() else np.nan
    # legacy (dance_live_eval / the earlier benchmark): instantaneous slope within 2 % among the
    # seconds with a tempo, phase vs the plain librosa reference over all beats after 20 s
    if "beats_old" in ref and len(ref["beats_old"]):
        g = np.gradient(th, t) * 60
        sec = np.arange(5, int(t[-1]))
        ii = np.searchsorted(t, sec)
        valid = has[ii]
        rb2 = np.interp(sec, ref["tempo_t"], ref["tempo_bpm"])
        r["legacy_level"] = 100 * float(np.mean(np.abs(g[ii][valid] / rb2[valid] - 1) < 0.02)) if valid.any() else 0.0
        bo = np.asarray(ref["beats_old"], float)
        bo = bo[(bo > 20) & (bo < t[-1] - 1)]
        eo = th_at(bo) - np.round(th_at(bo))
        eo_ms = np.abs(eo[:-1]) * np.diff(bo) * 1000
        r["legacy_med"], r["legacy_p90"] = float(np.median(eo_ms)), float(np.percentile(eo_ms, 90))
    if "cpu_ms" in L:
        r["cpu"] = float(np.mean(L["cpu_ms"]))
        r["cpu99"] = float(np.percentile(L["cpu_ms"], 99))
    return r


def _fmt(r, k, f="{:.0f}"):
    v = r.get(k, np.nan)
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def cmd_report(args):
    tr = tracks(args.refs, args.synth)
    dirs = args.dirs
    res = {}
    for d in dirs:
        res[d] = {}
        for key, (_, rp) in tr.items():
            f = Path(d) / f"{key}.npz"
            if f.exists():
                res[d][key] = evaluate(dict(np.load(f)), load_ref(rp))
    keys = [k for k in tr if all(k in res[d] for d in dirs)]
    cols = [("level %", "level", "{:.0f}"), ("tempo4 %", "tempo4", "{:.0f}"), ("lock s", "lock", "{:.1f}"),
            ("phase med", "med", "{:.0f}"),
            ("p90 ms", "p90", "{:.0f}"), ("off %", "off", "{:.1f}"), ("p90 mod 1/2", "p90_2x", "{:.0f}"),
            ("recov s", "recov", "{:.1f}"), ("flips", "flips", "{:.0f}"), ("conf r", "conf_r", "{:+.2f}"), ("AUC", "auc", "{:.2f}"),
            ("right %", "right", "{:.0f}")]
    hdr = "| track | " + " | ".join(c[0] for c in cols) + " |"
    print(hdr)
    print("|" + "---|" * (len(cols) + 1))
    for key in keys:
        cells = []
        for _, k, f in cols:
            vals = [_fmt(res[d][key], k, f) for d in dirs]
            cells.append(" -> ".join(vals) if len(set(vals)) > 1 or len(vals) == 1 else vals[0])
        print(f"| {key} | " + " | ".join(cells) + " |")
    for d in dirs:
        rr = [res[d][k] for k in keys]
        songs = [res[d][k] for k in keys if not k.startswith("synth")]
        agg = lambda name, ks: (name + ": " + ", ".join(  # noqa: E731
            f"{k} {np.nanmean([x.get(k, np.nan) for x in ks]):.2f}" for k in
            ("level", "tempo4", "lock", "med", "p90", "off", "p90_2x", "recov", "conf_r", "auc", "right", "legacy_level",
             "legacy_p90", "cpu", "cpu99")))
        print(agg(f"{d} mean (all)", rr))
        if songs:
            print(agg(f"{d} mean (songs)", songs))
        # confidence vs right over every reference beat of every track (pooled)
        for name, ks in (("all", keys), ("songs", [k for k in keys if not k.startswith("synth")])):
            c = np.concatenate([res[d][k]["_conf"] for k in ks])
            y = np.concatenate([res[d][k]["_right"] for k in ks]).astype(bool)
            if y.any() and (~y).any():
                from scipy.stats import mannwhitneyu
                auc = mannwhitneyu(c[y], c[~y]).statistic / (y.sum() * (~y).sum())
                print(f"{d} pooled confidence ({name}, {len(y)} beats, {100 * y.mean():.0f} % right): "
                      f"r {np.corrcoef(c, y)[0, 1]:+.2f}, AUC {auc:.2f}, mean conf right {c[y].mean():.2f} / wrong {c[~y].mean():.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps({d: {k: {m: v for m, v in r.items() if not m.startswith("_")}
                                                   for k, r in rr.items()} for d, rr in res.items()}, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="write the synthetic tracks (wav + reference npz)")
    s.add_argument("--dir", default=str(REF_DIR / "synth"))
    s = sub.add_parser("ref", help="build the offline references of the songs")
    s.add_argument("files", nargs="*", help="audio files (default: ~/Desktop/Anime/music/*.mp3)")
    s.add_argument("--out", default=str(REF_DIR))
    s.add_argument("--jobs", type=int, default=4)
    s.add_argument("--verbose", action="store_true")
    for name in ("run", "report"):
        s = sub.add_parser(name)
        s.add_argument("--refs", default=str(REF_DIR))
        s.add_argument("--synth", default=str(REF_DIR / "synth"))
        if name == "run":
            s.add_argument("--out", required=True)
            s.add_argument("--tempo", help="another tempo.py to benchmark (default: tools/dancelive/tempo.py)")
            s.add_argument("--only", nargs="*", help="substrings of the track keys")
            s.add_argument("--jobs", type=int, default=6)
        else:
            s.add_argument("dirs", nargs="+", help="run dirs (logs <key>.npz); two dirs = before -> after")
            s.add_argument("--json", help="save the metrics here")
    a = p.parse_args(argv)
    {"synth": cmd_synth, "ref": cmd_ref, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
