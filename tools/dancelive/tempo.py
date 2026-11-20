"""Tempo and beat clock: tempo estimate (onset autocorrelation + metrical comb, octave prior,
hysteresis), half-time decision, the beat-synchronous histogram PLL, and the beat-clock confidence.

TempoTracker is a mixin of analysis.LiveAnalyzer: it reads the analyzer's onset history
(self.onset, self.onset_low, self.n_onset, self.fps, self.dt, self.frames, self.max_dance_bpm) and
owns every tempo / phase / confidence attribute.

Beat clock (see _update_phase): three one-beat histograms of the onsets at the clock's own phase
(broadband: fine timing; broadband + kick / bass band, 4 s: which 16th / 8th position is the beat;
the same over ~25 s: the song's groove). The clock follows the broadband peak within +-0.1 beat
(it never slides to a neighbouring 16th); moving the beat to another position (the off-beat, a 16th,
a triplet) needs accumulated evidence: a CUSUM of the log evidence ratio on the slow histogram,
with a dead zone and up to 6 units of credit banked by the current position, or a fast path for a
clear contradiction (vetoed when the slow evidence backs the current position). Histogram memories
run on onset mass, not time: in a break (drums gone) they hold the phase, after the re-entry the
fresh onsets take over. The first lock leans on the kick (no history yet: kicks are on the beats in
most produced music); the long-term decision weighs it less (bass lines are often syncopated).
Tested with tools/tempo_bench.py (13 songs with offline references + synthetic tracks)."""
import math

import numpy as np

from .state import _ema


class TempoTracker:
    def _init_tempo(self, bpm_lo, bpm_hi, default_bpm):
        """Tempo / PLL / confidence parameters (needs self.fps and self.dt)."""
        self.a_conf = _ema(self.dt, 1.0)
        self.a_conf_out = _ema(self.dt, 2.5)
        self.a_str = _ema(self.dt, 8.0)
        self.str_slow = 0.0
        # tempo
        self.H = int(round(8.0 * self.fps))
        self.grid = np.arange(bpm_lo, bpm_hi + 1e-9, 0.25)
        self.lags = 60.0 * self.fps / self.grid
        # octave / metrical level: a person claps at the FASTEST comfortable pulse that is strongly
        # present. The comb score (beat + bar multiples) leans toward slow levels (73 vs 146, 99 vs 148
        # on real songs), so the candidates are weighted by tempo (linear tilt) up to 165 bpm.
        self.prior = (self.grid / 100.0) * (self.grid <= 165.0)
        self.min_tempo_score = 0.25
        self.tempo_every = max(1, int(round(0.25 * self.fps)))
        # a held tempo keeps an advantage: a challenger less than 30 % better must persist 4x longer,
        # and a weak periodicity (sparse / rubato passage) never moves it
        self.incumbent, self.min_switch_score = 0.3, 0.3
        self.settle = 15.0                       # ... once it has been held this long (s): the first pick stays mobile
        # the octave (x2 / /2) of the same beat grid is sticky: after the settle time, a move needs the
        # other octave to score 50 % better for 6 s (12 s on a confident lock); the move reinterprets
        # the grid (clock, histograms, lock and confidence carry over, the dance beat stays continuous)
        self.octave_margin, self.octave_hold = 0.5, 24
        # phase evidence: broadband flux + w_low x kick / bass flux, each relative to its running level
        self.NB = 48
        self.w_low = 0.6
        self.w_low_lock = 3.0                    # the first lock leans on the kick
        self.low_delay = 0.015                   # s: the low-band flux peaks ~15 ms after the onset (93 ms window)
        self.tau_h = 4.0                         # fast histograms: memory in s of normal onset density
        self.tau_s = 25.0                        # slow histogram (the groove)
        self.a_lvl = _ema(self.dt, 6.0)          # running onset levels (normalization)
        self.a_act_s, self.a_act_l = _ema(self.dt, 0.5), _ema(self.dt, 12.0)
        self.near = 5                            # fine tracking within +-5 bins (0.1 beat)
        self.w8 = 0.5                            # template: a beat comes with its 8th note (beat + 1/2)
        self.cus_d_slow, self.cus_h_slow = math.log(1.04), 2.0
        self.cus_credit, self.credit_rate = 6.0, 1.0
        self.cus_d_fast, self.cus_h_fast, self.fast_veto = math.log(1.5), 1.5, 2.0
        ph = np.arange(self.NB) / self.NB
        self.off_mask = np.abs(ph - 0.5) <= 0.1  # the off-beat: plain evidence (the template is symmetric there)
        self.near_mask = (ph <= 0.1) | (ph >= 0.9)
        self.lock_age = 2.5
        kb = np.minimum(np.arange(self.NB), self.NB - np.arange(self.NB)) / self.NB
        self.kernel_f = np.fft.rfft(np.exp(-0.5 * (kb / 0.03) ** 2))
        kb2 = np.minimum(np.arange(2 * self.NB), 2 * self.NB - np.arange(2 * self.NB)) / self.NB
        self.kernel2_f = np.fft.rfft(np.exp(-0.5 * (kb2 / 0.03) ** 2))
        self.k_phase = 0.03   # per hop: ~0.8 s time constant
        self.k_rate = 0.0004
        # rubato: the spread of the tempo estimates over ~20 s (log) -> u 0..1, which shortens the
        # histogram memory and frees the PLL (phase gain, rate gain and range) - the process noise of
        # the clock follows the music; steady songs keep u = 0
        self.a_tvar = 1.0 - math.exp(-1.0 / 80.0)  # per tempo update (every 0.25 s)
        self.rub_lo, self.rub_span = 0.015, 0.025
        self.rub_kp, self.rub_kr, self.rub_lim, self.rub_tau, self.rub_acf = 1.0, 40.0, 0.09, 0.6, 0.7
        self.default_rate = default_bpm / 60.0
        self.t_ref, self.theta_ref, self.rate = 0.0, 0.0, self.default_rate
        self.lvl_o = self.lvl_l = None

    def reset_tempo(self):
        self.onset = np.zeros(self.H)
        self.onset_low = np.zeros(self.H)
        self.n_onset = 0
        self.S = None
        self.music_bpm = None
        self.tempo_strength = 0.0
        self.pending, self.pending_bpm = 0, None
        self.tempo_age = 0.0
        self.integ = 0.0
        self.rubato = 0.0
        self.lb1 = self.lb2 = None
        self.locked = False
        self.R = 0.0
        self.conf = 0.0
        self.half = False
        self.off, self.div = 0.0, 1
        self.parity_bad = 0.0
        self.lvl_o = self.lvl_l = None
        self.act_s = self.act_l = 0.0
        self.reset_phase()

    def reset_phase(self):
        self.hist = np.zeros(self.NB)            # broadband onsets: fine timing
        self.hist_c = np.zeros(self.NB)          # broadband + kick, ~4 s: which position is the beat
        self.hist_s = np.zeros(self.NB)          # the same, ~25 s
        self.hist2 = np.zeros(2 * self.NB)       # two beats: the danced beat in half time
        self.phase_age = 0.0
        self.locked = False
        self.e_avg = 0.5
        self.beat_ratio = 1.0
        self.cusum = np.zeros(self.NB)
        self.cusum_f = np.zeros(self.NB)
        self.in_place = 0.0                      # s since the last lock / beat move
        self.margin = self.margin_f = self.Rf = 0.0

    # ---------------------------------------------------------------- clock helpers
    def theta(self, t):
        return self.theta_ref + (t - self.t_ref) * self.rate

    def _set_rate(self, rate, t):
        self.theta_ref, self.t_ref, self.rate = self.theta(t), t, rate

    def _set_div(self, div, t):
        """Change half time on/off keeping the dance beat continuous."""
        if div != self.div:
            th = self.theta(t)
            d = (th - self.off) / self.div
            self.off, self.div = th - d * div, div

    # ---------------------------------------------------------------- per hop
    def _tempo_hop(self, o, o_low, silence, t):
        """Onset history (newest last), tempo, PLL, then the confidence. Called once per hop."""
        if not silence:
            self.onset[:-1] = self.onset[1:]
            self.onset[-1] = o
            self.onset_low[:-1] = self.onset_low[1:]
            self.onset_low[-1] = o_low
            self.n_onset = min(self.n_onset + 1, self.H)
            self._levels(o, o_low)
            if self.n_onset >= 2.5 * self.fps and self.frames % self.tempo_every == 0:
                self._update_tempo(t)
            if self.music_bpm is not None:
                self._update_phase(t)
        if self.music_bpm is None:
            self.conf += (0.0 - self.conf) * self.a_conf
        else:
            # confidence ~ P(the clock is on the beat at the right level): a logistic model of the
            # clock's own evidence (phase concentration, jitter, the beat's margin over the other
            # positions, time since the last lock / move, a pending tempo change), fitted on the
            # benchmark (13 songs + synthetic tracks, leave-one-track-out AUC 0.89; the previous
            # product of terms: 0.76)
            self.str_slow += (self.tempo_strength - self.str_slow) * self.a_str
            c = 0.0
            if self.locked:
                z = (-4.7 + 5.4 * self.R + 3.6 * self.Rf - 13.3 * self.e_avg + 1.2 * self.margin_f
                     + 0.077 * min(self.in_place, 30.0) - (1.2 if self.pending > 0 else 0.0))
                c = 1.0 / (1.0 + math.exp(-min(max(z, -30.0), 30.0)))
            self.conf += (c - self.conf) * self.a_conf_out

    def _levels(self, o, o_low):
        """Running onset levels (to weigh the bands evenly) and the onset activity relative to the
        song's level: ~1 in normal music, << 1 in a break (no drums), > 1 just after it."""
        if self.lvl_o is None:
            self.lvl_o, self.lvl_l = max(o, 1e-3), max(o_low, 1e-3)
        self.lvl_o += (o - self.lvl_o) * self.a_lvl
        self.lvl_l += (o_low - self.lvl_l) * self.a_lvl
        v = o / (self.lvl_o + 1e-9) + o_low / (self.lvl_l + 1e-9)
        self.act_s += (v - self.act_s) * self.a_act_s
        self.act_l += (v - self.act_l) * self.a_act_l

    def rhythm(self):
        """1 when the music has a tempo, 0 while no steady pulse has been found."""
        return 1.0 if self.music_bpm else 0.0

    # ---------------------------------------------------------------- tempo
    def _update_tempo(self, t):
        x = self.onset[-self.n_onset:]
        x = x - x.mean()
        n = len(x)
        f = np.fft.rfft(x, 2 * n)
        ac = np.fft.irfft(f.real ** 2 + f.imag ** 2)[:n]
        if ac[0] <= 1e-12:
            return
        ac = ac / ac[0] * n / np.maximum(n - np.arange(n), 1)  # normalized, unbiased
        idx = np.arange(n)
        a1 = np.interp(self.lags, idx, ac)
        # metrical comb: the beat period, its multiples (bars) and its half (eighth notes) must all
        # correlate; a lag alone often prefers syncopated 3/16 or 5/16 note patterns
        score = (sum(np.maximum(np.interp(k * self.lags, idx, ac, right=0.0), 0.0) for k in (1, 2, 3, 4)) / 4.0
                 + 0.5 * np.maximum(np.interp(0.5 * self.lags, idx, ac), 0.0))
        self.S = score if self.S is None else self.S + (score - self.S) * 0.2  # ~1.2 s smoothing
        w = self.S * self.prior
        i = int(np.argmax(w))
        best = self._refine(ac, float(self.grid[i]))
        self.tempo_strength = float(max(a1[i], 0.0))
        if w[i] < self.min_tempo_score:  # no clear periodicity (rubato intro, ambient, speech)
            return
        self.tempo_age += self.tempo_every * self.dt
        if self.music_bpm is None:
            self.music_bpm = best
            self.integ = 0.0
            self.tempo_age = 0.0
        elif abs(best / self.music_bpm - 1.0) < 0.04:
            # the same tempo: smooth it (in rubato the PLL owns the swings, the 8 s autocorrelation
            # lags ~4 s), and measure its spread -> the rubato level
            self.music_bpm += 0.25 * (1.0 - self.rub_acf * self.rubato) * (best - self.music_bpm)
            self.pending = 0
            lb = math.log(self.music_bpm)
            if self.lb1 is None:
                self.lb1, self.lb2 = lb, lb * lb
            self.lb1 += (lb - self.lb1) * self.a_tvar
            self.lb2 += (lb * lb - self.lb2) * self.a_tvar
            sd = math.sqrt(max(self.lb2 - self.lb1 * self.lb1, 0.0))
            self.rubato = min(max((sd - self.rub_lo) / self.rub_span, 0.0), 1.0)
        else:  # a different tempo: switch only if it persists ~1.5 s (~3 s on a confident lock)
            j = int(np.argmin(np.abs(self.grid - self.music_bpm)))
            # the held tempo's own score, with the prior it was chosen with (refined above the 165 bpm
            # cut-off it would score 0 and lose to anything: the 168 <-> 84 flips)
            w_cur = float(self.S[max(j - 2, 0):j + 3].max()) * min(self.music_bpm, 165.0) / 100.0
            ratio = best / self.music_bpm
            octave = abs(ratio / 2.0 - 1.0) < 0.06 or abs(ratio * 2.0 - 1.0) < 0.06
            step, need = 1.0, (12 if self.conf > 0.5 else 6)
            if self.tempo_age > self.settle:  # a held tempo keeps an advantage
                if octave:
                    step = 1.0 if w[i] >= (1.0 + self.octave_margin) * w_cur else 0.0
                    need = self.octave_hold * (2 if self.conf > 0.5 else 1)
                else:
                    if w[i] < (1.0 + self.incumbent) * w_cur:
                        step = 0.25
                    if w[i] < self.min_switch_score:
                        step = 0.0
            if step == 0.0 and octave:
                self.pending = 0.0
            else:
                self.pending = self.pending + step if self.pending_bpm and abs(best / self.pending_bpm - 1) < 0.04 else step
            self.pending_bpm = best
            if self.pending >= need:
                if octave and self.locked:
                    self._reinterpret(best, t)  # the same beat grid at the other octave
                else:
                    self.music_bpm, self.pending, self.integ = best, 0, 0.0
                    self.tempo_age = 0.0
                    self.lb1 = self.lb2 = None  # a new tempo: its own rubato statistics
                    self.rubato = 0.0
                    self.reset_phase()  # re-acquire the phase at the new tempo
        thr = self.max_dance_bpm * (0.97 if self.half else 1.0)
        self.half = self.music_bpm > thr
        self._set_div(2 if self.half else 1, t)
        self._set_rate(self.music_bpm / 60.0 * (1.0 + self.integ), t)

    def _reinterpret(self, bpm, t):
        """Octave move (x2 or /2) of the same beat grid: the clock is rescaled (x2: each beat splits in
        two; /2: the new beat is the danced beat, or the stronger of each pair), the histograms are
        re-expressed on the new beat, the lock, the confidence and the danced beat carry over."""
        NB = self.NB
        th = self.theta(t)
        dance = (th - self.off) / self.div
        k = np.arange(NB)
        if bpm > self.music_bpm:  # one old beat = two new beats
            new_th = 2.0 * th
            hist2 = np.roll(np.repeat(self.hist_c, 2) / 2.0, -int(round(self.off * 2 * NB)))
            for name in ("hist", "hist_c", "hist_s"):  # both halves of the old beat land on the new beat
                h, g = getattr(self, name), np.zeros(NB)
                np.add.at(g, (2 * k) % NB, 0.5 * h)
                np.add.at(g, (2 * k + 1) % NB, 0.5 * h)
                setattr(self, name, g)
            self.hist2 = hist2
        else:  # two old beats = one new beat: hist2 (two old beats) is the new one-beat histogram
            h2 = self.hist2
            even = float(h2[:NB // 8].sum() + h2[-NB // 8:].sum())
            odd = float(h2[NB - NB // 8:NB + NB // 8].sum())
            first = 0.0 if (self.div == 2 or even >= odd) else 1.0  # the danced beat in half time
            new_th = (th - self.off - first) / 2.0
            g = np.roll(h2, -int(round(first * NB))).reshape(NB, 2).sum(1)
            hs, gs = self.hist_s, np.zeros(NB)  # the slow histogram has no parity: both halves
            gs += 0.5 * (hs[(2 * k) % NB] + hs[(2 * k + 1) % NB])
            self.hist, self.hist_c, self.hist_s = g * (self.hist.sum() / (g.sum() + 1e-12)), g, gs
            self.hist2 = np.tile(g, 2) / 2.0
        new_th += round(th - new_th)  # an integer shift keeps the beat count near its value (same grid)
        self.theta_ref, self.t_ref = new_th, t
        self.off = new_th - dance * self.div  # the danced beat is unchanged (_set_div then adapts div)
        self.music_bpm, self.pending, self.pending_bpm = bpm, 0, None
        self.lb1 = self.lb2 = None
        self.rubato = 0.0
        self.cusum[:] = 0.0
        self.cusum_f[:] = 0.0
        self.parity_bad = 0.0

    def _refine(self, ac, bpm):
        """Sub-frame tempo: the onset envelope is sampled at ~43 Hz, so the comb peaks sit on whole
        lags (95.7 vs 94.9 bpm at 27 / 109 frames). Fit a parabola around the peak near each
        multiple k*L and average the periods (long lags weigh more: they are more precise)."""
        L0 = 60.0 * self.fps / bpm
        num = den = 0.0
        for k in (1, 2, 3, 4):
            c = int(round(k * L0))
            if c + 3 >= len(ac):
                break
            j = c - 2 + int(np.argmax(ac[c - 2:c + 3]))
            y0, y1, y2 = ac[j - 1], ac[j], ac[j + 1]
            d = y0 - 2 * y1 + y2
            if y1 <= 0 or d >= 0:
                continue
            Lk = (j + 0.5 * (y0 - y2) / d) / k
            if abs(Lk / L0 - 1.0) < 0.03:
                num += k * y1 * Lk
                den += k * y1
        return 60.0 * self.fps / (num / den) if den > 0 else bpm

    # ---------------------------------------------------------------- phase-locked loop
    def _rotate(self, shift):
        """The clock phase moved by -shift beats: express the phase histograms in the new clock."""
        for name, nb in (("hist", self.NB), ("hist_c", self.NB), ("hist_s", self.NB), ("hist2", 2 * self.NB)):
            h = getattr(self, name)
            f = np.fft.rfftfreq(nb, 1.0 / nb)
            sh = -shift * nb / (nb / self.NB)  # in bins of that histogram (hist2 spans 2 beats)
            setattr(self, name, np.maximum(np.fft.irfft(np.fft.rfft(h) * np.exp(-2j * np.pi * f * sh / nb), nb), 0.0))

    def _peak_phase(self, score, i):
        """Phase (beats, -0.5..0.5) of the score peak at bin i, parabolic sub-bin."""
        NB = self.NB
        y0, y1, y2 = score[i - 1], score[i], score[(i + 1) % NB]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        return ((i + frac) / NB + 0.5) % 1.0 - 0.5

    def _template(self, sc):
        """A beat comes with its 8th note (beat + 1/2): rejects 16th / triplet positions."""
        return sc + self.w8 * np.roll(sc, -self.NB // 2)

    @staticmethod
    def _add(h, x, v):
        """Add v to the circular histogram h at phase x (0..1), linear between the two bins."""
        nb = len(h)
        u = x * nb
        i = int(u) % nb
        fr = u - int(u)
        h[i] += v * (1.0 - fr)
        h[(i + 1) % nb] += v * fr

    def _update_phase(self, t):
        """Beat-synchronous histogram PLL (see the module docstring)."""
        NB = self.NB
        a = self.onset[-1] / (self.lvl_o + 1e-9)
        b = self.onset_low[-1] / (self.lvl_l + 1e-9)
        r = min(max(self.act_s / (self.act_l + 1e-9), 0.1), 3.0)  # onset activity vs the song's level
        u = self.rubato
        decay = math.exp(-self.dt * r / (self.tau_h * (1.0 - self.rub_tau * u)))
        self.hist *= decay
        self.hist_c *= decay
        self.hist2 *= decay
        self.hist_s *= math.exp(-self.dt * r / self.tau_s)
        th = self.theta(t)
        thl = th - self.low_delay * self.rate
        self._add(self.hist, th % 1.0, a)
        self._add(self.hist_c, th % 1.0, a)
        self._add(self.hist_c, thl % 1.0, self.w_low * b)
        self._add(self.hist_s, th % 1.0, a)
        self._add(self.hist_s, thl % 1.0, self.w_low * b)
        self._add(self.hist2, ((th - self.off) % 2.0) / 2.0, a)
        self._add(self.hist2, ((thl - self.off) % 2.0) / 2.0, self.w_low * b)
        self.phase_age += self.dt * min(r, 1.0)
        if self.phase_age < self.lock_age:
            return
        fine = np.fft.irfft(np.fft.rfft(self.hist) * self.kernel_f, NB)
        score = np.fft.irfft(np.fft.rfft(self.hist_c) * self.kernel_f, NB)
        tmpl = self._template(score)
        sm = float(score.mean())
        i = int(np.argmax(tmpl))
        self.R = (score[i] - sm) / (score[i] + sm + 1e-12)  # 0 = flat, 1 = one sharp phase
        nr = self.near
        if not self.locked:
            if self.R > 0.15:  # the beat = the best template position (kick-heavy); timing from the broadband peak
                kick = np.fft.irfft(np.fft.rfft(self.hist_c - self.hist) * self.kernel_f, NB) / self.w_low
                i = int(np.argmax(self._template(fine + self.w_low_lock * kick)))
                w = (np.arange(i - nr, i + nr + 1)) % NB
                e = self._peak_phase(fine, int(w[np.argmax(fine[w])]))
                self.theta_ref -= e  # snap to the beats
                self._rotate(e)
                self.locked = True
                self.e_avg = 0.0
                self.cusum[:] = 0.0
                self.cusum_f[:] = 0.0
                self.in_place = 0.0
            return
        near = np.r_[NB - nr:NB, 0:nr + 1]
        s0, t0 = score[near].max(), tmpl[near].max()
        slow = np.fft.irfft(np.fft.rfft(self.hist_s) * self.kernel_f, NB)
        tslow = self._template(slow)
        sl0, tl0 = slow[near].max(), tslow[near].max()
        # is another position the beat? Log evidence ratio per bin (template ratio; plain ratio
        # around the off-beat, where the template is symmetric), accumulated and weighted by the onset
        # activity (a break accumulates nothing, the re-entry fast). Slow: dead zone +-delta (no clear
        # evidence either way accumulates nothing), credit banked by the current position (inertia
        # against a syncopated section). Fast: a clear contradiction, unless the groove disagrees.
        om = self.off_mask
        lr_s = np.where(om, np.log((slow + 1e-9) / (sl0 + 1e-9)), np.log((tslow + 1e-9) / (tl0 + 1e-9)))
        lr_f = np.where(om, np.log((score + 1e-9) / (s0 + 1e-9)), np.log((tmpl + 1e-9) / (t0 + 1e-9)))
        self.in_place += self.dt
        credit = min(self.cus_credit, self.credit_rate * self.in_place)
        dz = self.cus_d_slow
        inc = np.where(lr_s > dz, lr_s - dz, np.where(lr_s < -dz, lr_s + dz, 0.0))
        self.cusum = np.maximum(-credit, self.cusum + self.dt * r * inc)
        self.cusum_f = np.maximum(0.0, self.cusum_f + self.dt * r * (lr_f - self.cus_d_fast
                                                                     + self.fast_veto * np.minimum(lr_s, 0.0)))
        self.cusum[self.near_mask] = 0.0
        self.cusum_f[self.near_mask] = 0.0
        ks, kf = int(np.argmax(self.cusum)), int(np.argmax(self.cusum_f))
        best_q = (ks if self.cusum[ks] > self.cus_h_slow else
                  kf if self.cusum_f[kf] > self.cus_h_fast else None)
        self.margin = math.log((tl0 + 1e-9) / (np.where(self.near_mask, 0.0, tslow).max() + 1e-9))
        self.margin_f = math.log((t0 + 1e-9) / (np.where(self.near_mask, 0.0, tmpl).max() + 1e-9))
        fm, f0 = float(fine.mean()), float(fine[near].max())
        self.Rf = (f0 - fm) / (f0 + fm + 1e-12)
        self.beat_ratio = float(s0 / (score[om].max() + 1e-12))  # beat vs off-beat
        if best_q is not None:  # move the beat there (its timing from the broadband peak)
            w = (np.arange(best_q - nr, best_q + nr + 1)) % NB
            e = self._peak_phase(fine, int(w[np.argmax(fine[w])]))
            self.theta_ref -= e
            self._rotate(e)
            self.cusum[:] = 0.0
            self.cusum_f[:] = 0.0
            self.in_place = 0.0
            self.e_avg = 0.25
            return
        j = int(near[np.argmax(fine[near])])
        e = self._peak_phase(fine, j)
        d = self.k_phase * (1.0 + self.rub_kp * u) * e
        self.theta_ref -= d
        self._rotate(d)
        self.e_avg += (abs(e) - self.e_avg) * self.a_conf
        lim = 0.03 + self.rub_lim * u
        self.integ = min(max(self.integ - self.k_rate * (1.0 + self.rub_kr * u) * e, -lim), lim)
        self._set_rate(self.music_bpm / 60.0 * (1.0 + self.integ), t)
        if self.half:  # which musical beat is the danced one: the stronger of the two
            h2 = np.fft.irfft(np.fft.rfft(self.hist2) * self.kernel2_f, 2 * NB)
            even, odd = float(h2[:NB // 8].sum() + h2[-NB // 8:].sum()), float(h2[NB - NB // 8:NB + NB // 8].sum())
            if odd > 1.2 * even:
                self.parity_bad += self.dt
                if self.parity_bad > 2.0:
                    self.off -= 1.0  # dance beat +0.5
                    self.hist2 = np.roll(self.hist2, -NB)
                    self.parity_bad = 0.0
            else:
                self.parity_bad = max(0.0, self.parity_bad - self.dt)
