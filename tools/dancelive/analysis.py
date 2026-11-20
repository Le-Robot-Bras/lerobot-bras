"""Real-time analysis of the input audio, two rates: a fast hop (~5 ms) for what must react at once (band
levels for the accents, loudness, energy jumps, build-up / drop anticipation) and the tempo frame
(~23 ms, every `hop_div`-th fast hop, the same frames and the same ~93 ms window as ever) for onset
strength, the harmonic / percussive split, voice-band envelope, loudness / energy / sections and the DSP
features for the vocal student. The beat tracker (tempo.py) only sees the tempo frames: its input and
behaviour do not depend on the fast hop."""
import math
import threading
from collections import deque
from dataclasses import replace

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .drops import DropAnticipator
from .state import BeatState, _ema
from .tempo import TempoTracker

DSP_FEATURES = ("vox_db", "vox_env", "harm_ratio", "flatness", "bass", "mid", "high", "onset", "onset_low",
                "loud_db", "voice_band_db")


class LiveAnalyzer(TempoTracker):
    """Causal per-hop analysis of the input audio: spectrum, harmonic / percussive split, onset
    strength, band levels, voice-band envelope, loudness / energy / sections, the DSP feature rows
    (DSP_FEATURES) and the raw-audio history for the models; tempo + beat clock come from
    TempoTracker. process(x) per hop; snapshot() -> BeatState (thread-safe)."""
    BANDS = {"bass": (20, 150), "mid": (150, 2000), "high": (2000, 8000)}

    def __init__(self, sr, hop=None, max_dance_bpm=124.0, bpm_lo=70.0, bpm_hi=180.0, silence_db=-50.0,
                 default_bpm=90.0, loud_range=9.0, hop_div=4, accent_ms=46.0, band_attack=0.003,
                 band_release=0.15, loud_tau=3.0, fast_loud_tau=0.10, section_dwell=1.5, anticipate=True,
                 drop_dwell=2.5, drop_rank=0.80, drop_over_db=1.0, song_dist=3.5, song_gap=1.2, drop_max=45.0):
        """hop: the tempo frame (samples, default ~23 ms). hop_div: fast hops per tempo frame (1 = the
        old single-rate behaviour); process() takes `fast_hop` = hop / hop_div samples at a time (or
        any multiple, down to a whole `hop`). accent_ms: window of the band levels (the accents),
        band_attack / band_release: their envelope time constants (s). loud_tau: the loudness behind
        energy / sections (s); fast_loud_tau: the fast loudness (jump_db, energy_fast); section_dwell:
        s a section change must persist."""
        self.sr = sr
        self.hop = hop or (512 if sr < 32000 else 1024)
        d = max(1, int(hop_div))
        while self.hop % d:
            d -= 1
        self.hop_div, self.fast_hop = d, self.hop // d
        self.n_fft = 4 * self.hop
        self.fps = sr / self.hop              # tempo frames / s: what tempo.py's onset history runs at
        self.dt = 1.0 / self.fps
        self.fast_fps = sr / self.fast_hop
        self.fdt = 1.0 / self.fast_fps
        self.n_onset_bands = 64
        self.onset_shift = 0.028 * (self.n_fft / 2048.0) * (22050.0 / sr)
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.pnorm = 4.0 / float(self.window.sum()) ** 2  # full-scale sine -> power 1 (0 dB)
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / sr)
        # onset bands evenly spaced on the mel scale (like librosa's onset strength): on dense, loud
        # mixes this keeps the beat clearly above the off-beats; log spacing over-weights the bass
        mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
        edges = 700.0 * (10.0 ** (np.linspace(mel(30.0), mel(min(11000.0, 0.48 * sr)), self.n_onset_bands + 1)
                                  / 2595.0) - 1.0)
        M = np.stack([(freqs >= a) & (freqs < b) for a, b in zip(edges[:-1], edges[1:])]).astype(np.float32)
        keep = M.sum(axis=1) > 0
        self.M = M[keep]
        self.low = (np.sqrt(edges[:-1] * edges[1:])[keep] < 200.0)  # kick-drum bands
        self.bmask = {k: (freqs >= a) & (freqs < b) for k, (a, b) in self.BANDS.items()}
        # accent window: the band levels come from a shorter FFT than the tempo frames (the centre of the
        # Hann window is the delay of the level: 93 ms window -> 46 ms, 46 ms -> 23 ms)
        ns = 2 ** int(round(math.log2(max(256.0, accent_ms * 1e-3 * sr))))
        self.n_short = min(ns, self.n_fft)
        self.win_s = np.hanning(self.n_short).astype(np.float32)
        self.pnorm_s = 4.0 / float(self.win_s.sum()) ** 2
        fs = np.fft.rfftfreq(self.n_short, 1.0 / sr)
        self.Bm = np.stack([(fs >= a) & (fs < b) for a, b in self.BANDS.values()]).astype(np.float32)
        self.max_dance_bpm, self.silence_db, self.loud_range = max_dance_bpm, silence_db, loud_range
        self.a_flux = _ema(self.dt, 0.4)
        self.a_att, self.a_rel = _ema(self.fdt, band_attack), _ema(self.fdt, band_release)
        self.a_short, self.a_long = _ema(self.fdt, 0.4), _ema(self.fdt, loud_tau)
        self.a_fast = _ema(self.fdt, fast_loud_tau)
        self.section_dwell = section_dwell
        self.drop_dwell, self.drop_rank, self.drop_over_db = drop_dwell, drop_rank, drop_over_db
        self.song_dist_thr, self.song_gap = song_dist, song_gap
        self.drop_max, self.drop_since, self.drop_cool = drop_max, None, -1e9
        self._init_tempo(bpm_lo, bpm_hi, default_bpm)
        self.lock = threading.Lock()
        self.state = BeatState(rate=self.default_rate)
        self.buf = np.zeros(self.n_fft, np.float32)
        self.recent = np.zeros(int(8.0 * sr), np.float32)  # raw audio for the genre / vocal models
        self.recent_st = np.zeros((2, int(8.0 * sr)), np.float32)  # the same, stereo (Demucs separates wide choirs)
        self.recent_lock = threading.Lock()
        self.samples = 0
        self.frames = 0                       # tempo frames
        self.fframes = 0                      # fast hops
        self.reset_tempo()
        self.prev_db = None
        self.flux_mean = self.flux_low_mean = 0.0
        self.band_env = {k: -120.0 for k in self.BANDS}
        self.levels = {k: 0.0 for k in self.BANDS}
        self.lvl_hist = {k: deque(maxlen=int(10.0 * self.fps / 4)) for k in list(self.BANDS) + ["vox"]}
        self.lvl_range = {k: (-60.0, -30.0) for k in list(self.BANDS) + ["vox"]}
        self.mag_hist = np.zeros((9, self.n_fft // 2 + 1), np.float32)
        self.vox_db = -120.0
        self.vox_env = 0.0
        self.feat_rows = deque(maxlen=int(round(1.0 * self.fps)))  # last ~1 s of DSP features
        self.vmask = (freqs >= 300.0) & (freqs < 3400.0)
        self.a_vatt, self.a_vrel = _ema(self.dt, 0.02), _ema(self.dt, 0.08)
        self.percussive_onset = False
        self.p_short = self.p_long = self.p_fast = 0.0
        self.peak_long = None
        self.loud_hist = deque(maxlen=int(240.0 * self.fps / self.tempo_every))  # this song's loudness, 4 min
        # per-song statistics (reset at a song boundary): onset density and bass level, their typical values
        self.dens_s = 0.0
        self.dens_hist = deque(maxlen=int(240.0 / 0.25))
        self.bass_s = -120.0
        self.bass_hist = deque(maxlen=int(240.0 / 0.25))
        self.song_resets = []          # (stream time, "gap" | "profile") of every detected song boundary
        self.fp_hist = deque(maxlen=int(38.0 / 0.25))  # (spectral shape, level, tempo) every 0.25 s
        self._acc_g, self._acc_l, self._acc_n = np.zeros(16), 0.0, 0
        self.fp_dist = 0.0
        self.fp_over = 0
        self.song_age = 0.0
        self.sec_dense = False
        self.silent_for = 1e9
        self.silence = True
        self.section, self.sec_cand, self.sec_cand_t = "silence", "silence", 0.0
        self.energy = 0.0
        self.long_db = -120.0
        # fast events: loudness (tau fast_loud_tau) vs the level ~0.3-1.2 s earlier, rank in its own history
        self.lf_db = -120.0
        self.lf_hist = deque(maxlen=24)         # one value per 50 ms
        self.lf_every = max(1, int(round(0.05 * self.fast_fps)))
        self.lf_base = -120.0
        self.jump_db = 0.0
        self.lfd_hist = deque(maxlen=int(120.0 / 0.25))  # fast loudness every 0.25 s (2 min)
        self.lfd_every = int(round(0.25 * self.fast_fps))
        self.energy_fast = 0.0
        self.peak_fast = None
        self.anticipate = anticipate
        self.drops = DropAnticipator(frame_rate=self.fast_fps)
        self._o_last = 0.0
        self.t_frame = 0.0
        self.drop_eta = self.drop_t = self.drop_theta = None

    def process(self, x, stereo=None):
        """One fast hop (`fast_hop` samples); a longer block (e.g. a whole tempo hop) is fed hop by hop.
        x: the mono mix (everything is analysed on it); stereo: optional (2, n) of the same samples, kept
        only for recent_audio_stereo (default: the mono twice)."""
        fh = self.fast_hop
        if len(x) > fh:
            for i in range(0, len(x) - fh + 1, fh):
                self.process(x[i:i + fh], None if stereo is None else stereo[:, i:i + fh])
            return
        self.buf[:-fh] = self.buf[fh:]
        self.buf[-fh:] = x
        xs = np.stack([x, x]) if stereo is None else stereo
        with self.recent_lock:
            i = self.samples % len(self.recent)
            self.recent[i:i + fh] = x[:len(self.recent) - i]
            self.recent_st[:, i:i + fh] = xs[:, :len(self.recent) - i]
            if i + fh > len(self.recent):
                self.recent[:i + fh - len(self.recent)] = x[len(self.recent) - i:]
                self.recent_st[:, :i + fh - len(self.recent)] = xs[:, len(self.recent) - i:]
        self.samples += fh
        self.fframes += 1
        self._fast_hop(x)
        if self.fframes % self.hop_div == 0:
            self._tempo_frame()
        self._publish()

    # ---------------------------------------------------------------- fast path (every ~5 ms)
    def _fast_hop(self, x):
        # loudness: 0.4 s (silence), loud_tau (energy / sections), ~0.1 s (events)
        p = float(np.mean(x.astype(np.float64) ** 2))
        self.p_short += (p - self.p_short) * self.a_short
        if not self.silence:
            self.p_long += (p - self.p_long) * self.a_long
        self.p_fast += (p - self.p_fast) * self.a_fast
        self.lf_db = 10.0 * math.log10(self.p_fast + 1e-12)
        if self.fframes % self.lf_every == 0:
            if len(self.lf_hist) >= 12:  # the level 0.3 s .. 1.2 s ago
                self.lf_base = float(np.mean(list(self.lf_hist)[:-6]))
            self.lf_hist.append(self.lf_db)
        self.jump_db = self.lf_db - self.lf_base if self.lf_base > -100.0 and self.lf_db > -100.0 else 0.0
        # band levels for the accents: short window, fast attack, slow release (dB), each normalized on
        # its own distribution over the last ~10 s: 0 = its 10th percentile, 1 = its 97th (range >= 9 dB),
        # so a kick reads high and the gap after it low, on quiet and on loud, compressed masters alike
        X = np.fft.rfft(self.buf[-self.n_short:] * self.win_s)
        e = 10.0 * np.log10(self.Bm @ ((X.real ** 2 + X.imag ** 2) * self.pnorm_s) + 1e-12)
        for i, k in enumerate(self.BANDS):
            env = self.band_env[k]
            env += (float(e[i]) - env) * (self.a_att if e[i] > env else self.a_rel)
            self.band_env[k] = env
            self.levels[k] = self._norm_level(k, env, self.fframes)
        if self.anticipate:
            t = self.samples / self.sr
            clock = self.theta if (self.locked and self.music_bpm) else None
            self.drops.update(t, self.lf_db, self.band_env["bass"], self.band_env["high"], self._o_last, clock,
                              self.jump_db)

    # ---------------------------------------------------------------- tempo frame (every hop_div fast hops)
    def _tempo_frame(self):
        dt = self.dt
        self.frames += 1
        # centre of the analysis window, + the measured delay between an onset and its flux peak
        # (synthetic clicks: the peak comes ~28 ms early with a 2048-point Hann window at 22 kHz)
        t = (self.samples - self.n_fft / 2) / self.sr + self.onset_shift
        X = np.fft.rfft(self.buf * self.window)
        P = (X.real ** 2 + X.imag ** 2) * self.pnorm

        # causal harmonic / percussive split (soft masks): harmonic = median of this bin over the last
        # ~200 ms, percussive = median across ~180 Hz of frequency in this frame
        mag = np.sqrt(P)
        self.mag_hist[self.frames % len(self.mag_hist)] = mag
        Hm = np.median(self.mag_hist, axis=0)
        Pm = np.median(sliding_window_view(np.pad(mag, 8, mode="edge"), 17), axis=1)
        mh = Hm * Hm / (Hm * Hm + Pm * Pm + 1e-12)
        Ph = P * mh
        # voice-band harmonic energy -> fast envelope, normalized on its own recent distribution
        pv, phv = float(P[self.vmask].sum()), float(Ph[self.vmask].sum())
        ev = 10.0 * math.log10(phv + 1e-12)
        harm_ratio = phv / (pv + 1e-12)
        lp = np.log(P[self.vmask] + 1e-12)
        flatness = float(np.exp(lp.mean()) / (P[self.vmask].mean() + 1e-12))
        self.vox_db += (ev - self.vox_db) * (self.a_vatt if ev > self.vox_db else self.a_vrel)
        vox_env = self.vox_env = self._norm_level("vox", self.vox_db, self.fframes)

        # onset strength: spectral flux of log band levels, minus its running mean (the beat tracker
        # can use the percussive part only: sung syllables then pull it less off the beat)
        Po = P * (1.0 - mh) if self.percussive_onset else P
        db = np.maximum(10.0 * np.log10(self.M @ Po + 1e-12), -100.0)
        rise = np.maximum(db - self.prev_db, 0.0) if self.prev_db is not None else np.zeros_like(db)
        self.prev_db = db
        flux, flux_low = float(rise.mean()), float(rise[self.low].mean())
        self.flux_mean += (flux - self.flux_mean) * self.a_flux
        self.flux_low_mean += (flux_low - self.flux_low_mean) * self.a_flux
        o = max(0.0, flux - self.flux_mean)
        o_low = max(0.0, flux_low - self.flux_low_mean)
        self._o_last = o
        new_song = self._fingerprint(db, o, t)

        # loudness, silence, sections (the envelopes themselves run on the fast hops)
        short_db = 10.0 * math.log10(self.p_short + 1e-12)
        if short_db < self.silence_db:
            self.silent_for += dt
        else:
            if self.silent_for > 0.75:  # sound again: start the long envelopes where we are
                self.p_long = max(self.p_long, self.p_short)
                self.p_fast = max(self.p_fast, self.p_short)
                if self.silent_for >= self.song_gap:  # a gap between two songs
                    new_song = "gap"
            self.silent_for = 0.0
        silence = self.silence = self.silent_for > 0.75
        if self.silent_for > 3.0 and self.n_onset:  # music stopped: next song starts from scratch
            self.reset_tempo()
            self.drops.reset()
        if self.silent_for > 10.0:
            self.peak_long = self.peak_fast = None
            self.loud_hist.clear()
            self.lfd_hist.clear()
        if new_song:
            self._new_song(t, new_song)
        long_db = self.long_db = 10.0 * math.log10(self.p_long + 1e-12)
        energy, below = 0.0, 0.0
        if not silence:
            if self.frames % self.tempo_every == 0:
                self.loud_hist.append(long_db)
                self.lfd_hist.append(self.lf_db)
            if self.peak_long is None:  # no history: a typical modern master is loud at about -6 dBFS
                self.peak_long = max(long_db + 3.0, -6.0)
            self.peak_long = max(long_db, self.peak_long - 0.1 * dt)
            if len(self.loud_hist) >= 15.0 / (self.tempo_every * dt):
                # the song's own running loudness distribution (last 2 min): percentile rank of now,
                # and how far below its loud level (90th percentile)
                h = np.fromiter(self.loud_hist, float)
                energy = float(np.mean(h < long_db))
                below = float(np.percentile(h, 90)) - long_db
                self._set_drop_evidence(long_db - float(np.median(h)))
            else:  # first seconds of a song: relative to an absolute anchor, never a drop
                below = self.peak_long - long_db
                energy = min(max(1.0 - below / self.loud_range, 0.0), 1.0)
                self.sec_drop_ok = False
            # the fast twin of energy: the 0.1 s loudness ranked in its own 2 min distribution
            if self.peak_fast is None:
                self.peak_fast = max(self.lf_db + 3.0, -6.0)
            self.peak_fast = max(self.lf_db, self.peak_fast - 0.1 * dt)
            if len(self.lfd_hist) >= 60:
                self.energy_fast = float(np.mean(np.fromiter(self.lfd_hist, float) < self.lf_db))
            else:
                self.energy_fast = min(max(1.0 - (self.peak_fast - self.lf_db) / self.loud_range, 0.0), 1.0)
        else:
            self.energy_fast = 0.0
        self.energy = energy
        self._update_section(silence, energy, below, t)

        # onset history (newest last), tempo, PLL, confidence
        self._tempo_hop(o, o_low, silence, t)

        with self.recent_lock:
            self.feat_rows.append((self.vox_db, vox_env, harm_ratio, flatness, self.levels["bass"], self.levels["mid"],
                                   self.levels["high"], o, o_low, long_db, 10.0 * math.log10(pv + 1e-12)))
        self.t_frame = t

    def _publish(self):
        """BeatState of the latest fast hop: the tempo-frame values + the fast ones."""
        t_audio = self.samples / self.sr
        theta_d = None
        if self.anticipate and self.music_bpm:
            theta_d = self.drops.eta_theta(self.theta(t_audio))
        if theta_d is not None:
            self.drop_theta = theta_d
            self.drop_t = self.t_ref + (theta_d - self.theta_ref) / self.rate
            self.drop_eta = max(self.drop_t - t_audio, 0.0)
        else:
            self.drop_theta = self.drop_t = self.drop_eta = None
        L = self.levels
        st = BeatState(t=self.t_frame, t_ref=self.t_ref, theta_ref=self.theta_ref, rate=self.rate,
                       off=self.off, div=self.div, music_bpm=self.music_bpm or 0.0,
                       bpm=(self.rate * 60.0 / self.div) if self.music_bpm else 0.0, conf=self.conf,
                       locked=self.locked, bass=L["bass"], mid=L["mid"], high=L["high"],
                       loud_db=self.long_db, energy=self.energy, section=self.section, vox_env=self.vox_env,
                       t_audio=t_audio, loud_fast_db=self.lf_db, jump_db=self.jump_db,
                       energy_fast=self.energy_fast, build_level=self.drops.level if self.anticipate else 0.0,
                       drop_eta=self.drop_eta, drop_t=self.drop_t, drop_theta=self.drop_theta)
        with self.lock:
            self.state = st

    def _norm_level(self, k, db, n):
        """0..1 position of db in the recent distribution of this band (percentiles; appended every 4th
        tempo frame, range refreshed every 8th: `n` counts fast hops so the cadence does not depend on
        the hop split)."""
        h = self.lvl_hist[k]
        step = self.hop_div
        if n % (4 * step) == 0 and db > -95.0:
            h.append(db)
        if n % (8 * step) == 0 and len(h) > 10:
            lo, hi = np.percentile(np.fromiter(h, float), (10, 97))
            self.lvl_range[k] = (lo, max(hi, lo + 9.0))
        lo, hi = self.lvl_range[k]
        return min(max((db - lo) / (hi - lo), 0.0), 1.0)

    def snapshot(self):
        with self.lock:
            return replace(self.state)

    def feature_vector(self):
        """DSP features of the last ~1 s for the vocal / instrument student: mean and std of the
        per-hop rows (see DSP_FEATURES)."""
        with self.recent_lock:
            r = np.array(self.feat_rows, np.float64) if self.feat_rows else np.zeros((1, len(DSP_FEATURES)))
        return np.concatenate([r.mean(0), r.std(0)])

    def recent_audio(self, seconds):
        """The last `seconds` of input audio (oldest first), or None if not that much yet."""
        n = int(seconds * self.sr)
        with self.recent_lock:
            if self.samples < n:
                return None
            i = self.samples % len(self.recent)
            return np.roll(self.recent, -i)[-n:].copy()

    def recent_audio_stereo(self, seconds):
        """The last `seconds` of input audio as float32 (2, n) (oldest first, same alignment as recent_audio;
        a mono input appears as the same channel twice), or None if not that much yet."""
        n = int(seconds * self.sr)
        with self.recent_lock:
            if self.samples < n:
                return None
            i = self.samples % self.recent.shape[0]
            return np.roll(self.recent_st, -i, axis=1)[:, -n:].copy()

    def mark_eof(self):
        with self.lock:
            self.state = replace(self.state, eof=True, section="silence")

    # ---------------------------------------------------------------- sections
    def _fingerprint(self, db, o, t):
        """Per tempo frame: this song's onset density and bass level, and the song-boundary test for songs that
        follow each other without a gap. Three cues, each compared over the last 8 s against the 30-40 s
        before: (P) the spectral shape (16 mel groups, dB) moved by more than song_dist, (L) the mean level
        moved by more than 3 dB, (T) the tempo moved by more than 6 %. Two of the three for 2 s = a new song
        ("profile"). Returns "profile" or None."""
        if self.silent_for > 0.0 or db is None:
            return None
        g = np.array([x.mean() for x in np.array_split(db, 16)])
        self._acc_g += g
        self._acc_l += float(db.mean())
        self._acc_n += 1
        self.dens_s += (o - self.dens_s) * _ema(self.dt, 3.0)
        be = self.band_env["bass"]
        self.bass_s = be if self.bass_s < -100.0 else self.bass_s + (be - self.bass_s) * _ema(self.dt, 3.0)
        if self.frames % self.tempo_every:
            return None
        self.dens_hist.append(self.dens_s)
        self.bass_hist.append(self.bass_s)
        n = max(self._acc_n, 1)
        self.fp_hist.append((self._acc_g / n, self._acc_l / n, self.music_bpm or 0.0))
        self._acc_g = np.zeros(16)
        self._acc_l, self._acc_n = 0.0, 0
        self.song_age += 0.25
        if len(self.fp_hist) < self.fp_hist.maxlen:
            return None
        G = np.array([x[0] for x in self.fp_hist])
        Lv = np.array([x[1] for x in self.fp_hist])
        Tp = np.array([x[2] for x in self.fp_hist])
        A = 32  # 8 s
        a, b = G[-A:].mean(0), G[:-A].mean(0)
        self.fp_dist = float(np.sqrt(np.mean(((a - a.mean()) - (b - b.mean())) ** 2)))
        p = self.fp_dist > self.song_dist_thr
        lv = abs(float(Lv[-A:].mean()) - float(np.median(Lv[:-A]))) > 3.0
        ta, tb = float(np.median(Tp[-A:])), float(np.median(Tp[:-A]))
        tp = ta > 0 and tb > 0 and abs(ta / tb - 1.0) > 0.06
        self.fp_over = self.fp_over + 1 if (int(p) + int(lv) + int(tp) >= 2 and self.song_age > 35.0) else 0
        return "profile" if self.fp_over >= 8 else None

    def _new_song(self, t, why):
        """A new song starts: its loudness / density / bass statistics start from scratch (after a gap) or
        from the last ~20 s (a boundary found without a gap: the new song has been playing for that long)."""
        self.song_resets.append((t, why))
        keep = 0 if why == "gap" else int(20.0 / 0.25)
        for h in (self.loud_hist, self.lfd_hist, self.dens_hist, self.bass_hist):
            tail = list(h)[-keep:] if keep else []
            h.clear()
            h.extend(tail)
        self.fp_hist.clear()
        self.peak_long = self.peak_fast = None
        self.p_long = self.p_fast = self.p_short
        self.fp_over, self.song_age, self.fp_dist = 0, 0.0, 0.0
        if why == "gap":
            self.dens_s, self.bass_s = 0.0, -120.0
        self.section = self.sec_cand = "normal" if self.section != "silence" else "silence"
        self.sec_cand_t = t
        self.sec_drop_ok = False

    def _set_drop_evidence(self, over_db):
        """drop = clearly louder than this song's typical level AND denser (more onsets or more bass than
        its typical). Called with how far the loudness is above the song's median (dB)."""
        dh, bh = self.dens_hist, self.bass_hist
        dense = False
        if len(dh) >= 40:
            dense = (self.dens_s > 1.10 * float(np.median(dh)) or self.bass_s > float(np.median(bh)) + 1.0)
        self.sec_over_db, self.sec_dense = over_db, dense
        self.sec_drop_ok = True

    def _update_section(self, silence, energy, below, t):
        """calm: clearly under the song's loud level (>= 3.5 dB) and in its quietest quarter.
        drop: in the loudest ~22 % of THIS song's loudness (rank >= drop_rank) AND >= drop_over_db above its
        median AND denser than its typical (onset density / bass), for drop_dwell s. Normal otherwise.
        Hysteresis on both; calm / normal changes must persist section_dwell s."""
        if silence:
            self.section = self.sec_cand = "silence"
            return
        cur = self.section
        h_calm = (0.5, 0.05) if cur == "calm" else (0.0, 0.0)
        in_drop = cur == "drop"
        over = getattr(self, "sec_over_db", 0.0)
        ok = getattr(self, "sec_drop_ok", False)
        rank_thr = self.drop_rank - (0.12 if in_drop else 0.0)
        over_thr = self.drop_over_db - (0.5 if in_drop else 0.0)
        if below > 3.5 - h_calm[0] and energy < 0.25 + h_calm[1]:
            cand = "calm"
        elif ok and energy >= rank_thr and over >= over_thr and (self.sec_dense or in_drop) and t >= self.drop_cool:
            cand = "drop"
        else:
            cand = "normal"
        if cur == "silence":  # music starts: no dwell
            self.section = self.sec_cand = "normal" if cand == "drop" else cand
            self.sec_cand_t = t
            return
        if cand != self.sec_cand:
            self.sec_cand, self.sec_cand_t = cand, t
        dwell = max(self.section_dwell, self.drop_dwell) if cand == "drop" else self.section_dwell
        if cand != cur and t - self.sec_cand_t >= dwell:
            self.section = cand
            self.drop_since = t if cand == "drop" else None
        if self.section == "drop" and self.drop_since is not None and t - self.drop_since > self.drop_max:
            # a "drop" that lasts this long is the song's normal level: back to normal for a while
            self.section, self.sec_cand, self.sec_cand_t = "normal", "normal", t
            self.drop_since, self.drop_cool = None, t + 20.0
