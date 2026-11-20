"""Drop anticipation: recognise a build-up while it happens and predict where the drop will land.

A build-up (EDM riser, snare roll, a break that gathers energy, a band falling silent before the
whole band re-enters) shows as: loudness and / or the high band (2-8 kHz, noise risers) rising
over the last seconds, the bass / kick dropping out (the song's loud level is no longer reached),
and the onset density growing (snare roll). The detector turns these into build_level (0..1).

The drop lands on a bar line of the beat clock, and builds are 1, 2, 4, 8 or 16 bars long: the
build started on a bar line, so the drop is at its start + 4, 8, 16, 32 or 64 musical beats. The
candidate chosen is the smallest one that is still ahead of "now"; a short hard silence ("gap",
the last beat of many builds) pins it: the drop is the first beat after the gap.

DropAnticipator is causal and cheap: update() once per analysis frame, a full evaluation every
0.25 s on a 0.25 s-resolution history of loudness / bass / high levels and onset strength.
Outputs: level (build_level), theta (musical beat number of the predicted drop, or None). The
analyzer converts theta to seconds with its own beat clock (so it follows tempo corrections)."""
import math
from collections import deque

import numpy as np

from .state import _ema

CANDIDATE_BEATS = (8, 16, 32, 48, 64, 80, 96, 128)  # build length from the break / build start: 2 .. 32 bars
PRIOR = {8: 0.06, 16: 0.22, 32: 0.36, 48: 0.06, 64: 0.18, 80: 0.04, 96: 0.03, 128: 0.05}  # 8 bars is the
# standard build, then 4 and 16; a song repeats its own (song_N)
STEP = 0.25                           # s, resolution of the history
MIN_PUBLISH_ELAPSED = 6.0             # beats into the build before a drop is announced


def _slope(x, dt):
    """Least-squares slope of the samples x (dB per second)."""
    n = len(x)
    if n < 3:
        return 0.0
    k = np.arange(n) - 0.5 * (n - 1)
    return float(np.dot(k, np.asarray(x) - np.mean(x)) / (np.dot(k, k) * dt))


class DropAnticipator:
    def __init__(self, frame_rate=172.0, hist_s=64.0, publish_beats=20.0, start_level=0.35, end_level=0.18):
        self.frame_rate = frame_rate
        self.n_hist = int(hist_s / STEP)
        self.publish_beats = publish_beats
        self.start_level, self.end_level = start_level, end_level
        self.a_up, self.a_down = _ema(STEP, 0.6), _ema(STEP, 1.2)
        self.reset()

    def reset(self):
        self.loud, self.bass, self.high = (deque(maxlen=self.n_hist) for _ in range(3))
        self.onset = deque(maxlen=self.n_hist)
        self.braw = deque(maxlen=self.n_hist)
        self.rise_h, self.struct_h = deque(maxlen=self.n_hist), deque(maxlen=self.n_hist)
        self.song_N = None   # the build length (beats) of the last drop of this song: builds repeat
        self.acc = np.zeros(4)
        self.n_acc = 0
        self.t_next = None
        self.level = 0.0
        self.active = False
        self.t_start = None
        self.t_anchor = None
        self.theta = None
        self.max1s = deque(maxlen=int(self.frame_rate))  # fast loudness over the last ~1 s (gap detection)
        self.ref_l = None
        self.gap_theta = None
        self.last_drop_t = -1e9
        self.n_over = 0

    # ---------------------------------------------------------------- per analysis frame
    def update(self, t, loud_db, bass_db, high_db, onset, clock, jump_db=0.0):
        """t: stream time (s); *_db: fast levels; onset: onset strength; clock: theta(t) -> musical
        beat, or None while there is no locked tempo; jump_db: the loudness step (analysis.jump_db)."""
        self.acc += (loud_db, bass_db, high_db, onset)
        self.n_acc += 1
        self.max1s.append(loud_db)
        if self.t_next is None:
            self.t_next = t + STEP
        if clock is not None and self.ref_l is not None and len(self.max1s) > 8 and (self.active or self.level > 0.2):
            # gap: the level is far under the song's loud level and has just fallen (the "breath" / the
            # bottom of the break before the drop): the drop is the next beat; as long as the gap lasts
            # the prediction slides to the next beat
            if loud_db < self.ref_l - 9.0 and loud_db < max(self.max1s) - 6.0 and loud_db > -90.0:
                self.gap_theta = clock(t)
            elif self.gap_theta is not None and loud_db > self.ref_l - 6.0:
                self.gap_theta = None
        if (self.active or self.gap_theta is not None) and jump_db > 5.0 and clock is not None:
            self._drop_seen(t, clock)  # a hard energy step up: the drop has landed
        if t < self.t_next:
            return
        m = self.acc / max(self.n_acc, 1)
        self.acc[:] = 0.0
        self.n_acc = 0
        self.t_next = t + STEP
        self.loud.append(m[0])
        self.bass.append(m[1])
        self.high.append(m[2])
        self.onset.append(m[3])
        self._evaluate(t, clock)

    def _drop_seen(self, t, clock):
        """The drop landed (~0.15 s ago: the jump needs that long to show). Learn this song's build length."""
        if self.t_anchor is not None:
            n = (clock(t - 0.15) - clock(self.t_anchor))
            best = min(CANDIDATE_BEATS, key=lambda c: abs(c - n))
            if abs(best - n) <= 3.0:
                self.song_N = best
        self.last_drop_t = t
        self.active = False
        self.theta = self.gap_theta = None
        self.level *= 0.3

    # ---------------------------------------------------------------- every 0.25 s
    def _evaluate(self, t, clock):
        n = len(self.loud)
        if n < 60:  # needs ~15 s of music to know the song's loud level
            self.braw.append(0.0)
            return
        loud, bass, high = np.array(self.loud), np.array(self.bass), np.array(self.high)
        ref_l, ref_b = np.percentile(loud, 85), np.percentile(bass, 85)
        self.ref_l = ref_l
        l1, b1 = loud[-4:].mean(), bass[-4:].mean()
        r_loud = min(max(_slope(loud[-12:], STEP) / 1.2, 0.0), 1.0)
        r_high = min(max(_slope(high[-12:], STEP) / 2.0, 0.0), 1.0)
        dip = min(max((ref_l - l1 - 1.0) / 6.0, 0.0), 1.0)
        kick_out = min(max((ref_b - b1 - 3.0) / 9.0, 0.0), 1.0)
        on = np.array(self.onset)
        ratio = on[-8:].mean() / (on[-40:-8].mean() + 1e-9)
        dens = min(max((ratio - 1.3) / 1.2, 0.0), 1.0)
        rising, structure = max(r_loud, r_high), max(dip, kick_out)
        self.rise_h.append(rising)
        self.struct_h.append(structure)
        b_raw = min(0.45 * rising + 0.30 * structure + 0.25 * dens + 0.20 * rising * structure, 1.0)
        if t - self.last_drop_t < 3.0:
            b_raw = 0.0
        self.braw.append(b_raw)
        a = self.a_up if b_raw > self.level else self.a_down
        self.level += (b_raw - self.level) * a
        if not self.active:
            self.n_over = self.n_over + 1 if self.level > self.start_level else 0
            if self.n_over >= 3 and clock is not None:
                self.active = True
                # the build began where the evidence first appeared (walk back through the history)
                k = 0
                for v in reversed(self.braw):
                    if v < 0.12 or k > 80:
                        break
                    k += 1
                self.t_start = t - k * STEP
                self.theta = None
                self.gap_theta = None
                self.t_anchor = None
        elif self.level < self.end_level or t - self.t_start > 80.0:
            self.active = False
            self.theta = self.gap_theta = None
        self.theta = None
        if clock is not None and (self.active or self.gap_theta is not None):
            th_now = clock(t)
            cand = self._predict(t, clock) if self.active else None
            if self.gap_theta is not None:  # the breath before the drop: the drop is the next beat
                g = math.ceil(th_now + 0.3)
                self.theta = float(cand if cand is not None and abs(cand - g) <= 1.0 and cand >= th_now - 0.5 else g)
            else:
                self.theta = cand

    def _run_start(self, series, thr, t, lag, max_gap=4):
        """Stream time at which the contiguous run (gaps < 1 s) of series > thr that reaches now began,
        minus the detection lag of that evidence; None if there is no such run."""
        k, gap, first = 0, 0, None
        for v in reversed(series):
            if v > thr:
                first, gap = k, 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            k += 1
            if k > 300:
                break
        return None if first is None else t - first * STEP - lag

    def _predict(self, t, clock):
        th_now = clock(t)
        # the build proper began where the evidence of a build began: a riser (rising) that starts well
        # after the break (structure: kick / bass out, level down) is the build, the break was before it
        t_s = self._run_start(self.struct_h, 0.25, t, 0.5)
        t_r = self._run_start(self.rise_h, 0.20, t, 1.0, max_gap=8)
        starts = [x for x in (t_s, t_r) if x is not None]
        if not starts:
            return None
        t_a = max(min(starts), self.t_start - 1.0)
        # the anchor is the break / build start (kick or level falling away: an abrupt, well localised
        # event) and stays where it was first put, whatever the slope estimates do afterwards
        if self.t_anchor is None:
            self.t_anchor = t_a
        th_a = round(clock(self.t_anchor))
        e = th_now - th_a
        # candidates still ahead (kept 1 beat after they pass: the drop may be a beat late); the prior favours
        # the usual 8 bars and this song's own build length
        ahead = [n for n in CANDIDATE_BEATS if th_a + n >= th_now - 1.0] or [CANDIDATE_BEATS[-1]]
        w = {n: PRIOR[n] * (3.0 if n == self.song_N else 1.0) for n in ahead}
        cand = th_a + max(w, key=w.get)
        if self.song_N is None or e < MIN_PUBLISH_ELAPSED or e < 0:
            return None  # the grid guess is only announced once this song's own build length is known
        return float(cand)

    # ---------------------------------------------------------------- read out
    def eta_theta(self, th_now):
        """The predicted drop (musical beat number) if it is near enough to be useful, else None."""
        if self.theta is None or self.theta - th_now > self.publish_beats or self.theta - th_now < -1.5:
            return None
        return self.theta
