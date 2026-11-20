"""Vocals and instruments: the trained student (tools/train_vocal_model.py, ~/.cache/so101/vocal_model.npz:
v2 = logistic regressions on YAMNet's 1024-d embedding + vocal_dsp() every 0.25 s, "lite" head on
YAMNet scores when no embedding; v1 files still load), the Demucs vocal gate (with YAMNet on the
separated vocal stem), the fusion of both (vocal_gate) and the per-tick vocal decision + mouth
(VocalFusion)."""
import math
import threading
import time
from pathlib import Path

import numpy as np

from .analysis import DSP_FEATURES  # noqa: F401  (the student's DSP feature names, re-exported)
from .state import _ema

# ================================================================ vocals / instruments (student)
STUDENT_TARGETS = ("vocals", "drums", "bass", "other")
STUDENT_PATH = Path.home() / ".cache/so101/vocal_model.npz"
STUDENT_VERSION = 2  # 1: logistic on YAMNet scores + analyzer DSP (still loadable), 2: heads full / lite
FAMILIES = {  # YAMNet instrument classes -> family tag shown / used by the dance
    "GTR": ("Electric guitar", "Guitar", "Distortion", "Power chord" , "Bass guitar"),
    "SYNTH": ("Synthesizer", "Sampler", "Drum machine", "Electronic organ"),
    "PIANO": ("Piano", "Electric piano", "Keyboard (musical)"),
    "STRINGS": ("Orchestra", "String section", "Violin, fiddle", "Cello", "Bowed string instrument", "Harp"),
    "BRASS": ("Brass instrument", "Trumpet", "Trombone", "French horn", "Saxophone"),
    "CHOIR": ("Choir", "Chant"),
}


def student_vector(yam, dsp):
    """v1 student features: log YAMNet scores + the analyzer's DSP feature vector."""
    return np.concatenate([np.log(np.asarray(yam, np.float64) + 1e-3), dsp])


def student_feature_names(yam_names):
    return ([f"yamnet:{n}" for n in yam_names] + [f"mean:{n}" for n in DSP_FEATURES]
            + [f"std:{n}" for n in DSP_FEATURES])


# ---------------------------------------------------------------- v2 student features
# Self-contained (raw audio in, no dependency on the analyzer's features): spectral shape / voice cues
# of the last 1 s, resampled to 22.05 kHz, 1024-pt frames, hop 512 (~42 frames).
VDSP_SR, _VN, _VH = 22050, 1024, 512
_vf = np.fft.rfftfreq(_VN, 1.0 / VDSP_SR)
_vm = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
_ve = 700.0 * (10.0 ** (np.linspace(_vm(40.0), _vm(10000.0), 33) / 2595.0) - 1.0)
_VMEL = np.stack([np.clip(np.minimum((_vf - a) / (b - a), (c - _vf) / (c - b)), 0.0, None)
                  for a, b, c in zip(_ve[:-2], _ve[1:-1], _ve[2:])])  # 31 triangular mel filters
_VBAND = (_vf >= 300.0) & (_vf < 3400.0)
_VWIN = np.hanning(_VN)
_VROWS = ("flat", "peak", "flux", "cent", "vox_rel", "cont")
VOCAL_DSP_NAMES = ([f"mel{i}" for i in range(31)] + [f"mel{i}_sd" for i in range(31)] + list(_VROWS)
                   + [f"{n}_sd" for n in _VROWS] + ["lvl", "lvl_sd", "mod"])


def vocal_dsp(x, sr):
    """v2 DSP features of audio x (the last ~1 s, any rate): mel spectral shape (31 bands, level
    removed) mean / std; voice band (300-3400 Hz) flatness, spectral-peak energy share (tonal
    partials, any number of simultaneous pitches: choirs), flux, centroid, voice-band share,
    frame-to-frame spectral continuity (held / gliding notes) mean / std; level; 3-9 Hz modulation
    of the voice band (syllable rate). 77 values, ~5 ms."""
    x = np.asarray(x, np.float64)
    if sr != VDSP_SR:
        n = int(round(len(x) * VDSP_SR / sr))
        x = np.fft.irfft(np.fft.rfft(x)[:n // 2 + 1], n) * (n / len(x))
    if len(x) < 2 * _VN:
        x = np.pad(x, (2 * _VN - len(x), 0))
    fr = np.lib.stride_tricks.sliding_window_view(x, _VN)[::_VH] * _VWIN
    S = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-10
    M = 10.0 * np.log10(S @ _VMEL.T + 1e-10)
    lvl = M.mean(1, keepdims=True)
    shape = M - lvl
    Sv = S[:, _VBAND]
    flat = np.exp(np.mean(np.log(Sv), 1)) / np.mean(Sv, 1)
    loc = np.maximum(np.roll(Sv, 1, 1), np.roll(Sv, -1, 1))
    pk = (Sv >= loc) & (Sv > 2.0 * (np.roll(Sv, 3, 1) + np.roll(Sv, -3, 1)))
    peak = (Sv * pk).sum(1) / Sv.sum(1)
    flux = np.r_[0.0, np.maximum(np.diff(M, axis=0), 0.0).mean(1)]
    cen = (S * _vf).sum(1) / S.sum(1) / 1000.0
    ve = 10.0 * np.log10(Sv.sum(1))
    ve_rel = ve - 10.0 * np.log10(S.sum(1))
    env = ve - ve.mean()
    sp = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    fm = np.fft.rfftfreq(len(env), _VH / VDSP_SR)
    mod = sp[(fm >= 3) & (fm <= 9)].sum() / (sp[fm > 0.5].sum() + 1e-9)
    lv = np.log(Sv)
    lv = lv - lv.mean(1, keepdims=True)
    nv = np.sqrt((lv * lv).sum(1)) + 1e-12
    cont = np.r_[0.0, (lv[1:] * lv[:-1]).sum(1) / (nv[1:] * nv[:-1])]
    rows = np.stack([flat, peak, flux, cen, ve_rel, cont], 1)
    return np.concatenate([shape.mean(0), shape.std(0), rows.mean(0), rows.std(0),
                           [lvl.mean(), lvl.std(), mod]]).astype(np.float32)


class _Head:
    """One v2 student head (features -> 4 probabilities): standardization + per-target logistic
    regression ("lin") or one-hidden-layer ReLU MLP ("mlp"), all numpy."""

    def __init__(self, d, h):
        self.name = h
        self.kind = str(d[f"{h}_kind"])
        self.mu, self.sd = d[f"{h}_mu"], d[f"{h}_sd"]
        self.P = {k[len(h) + 1:]: d[k] for k in d.files if k.startswith(h + "_") and k not in (f"{h}_kind",)}

    def __call__(self, f):
        z = (np.asarray(f, np.float64) - self.mu) / self.sd
        P = self.P
        if self.kind == "lin":
            s = P["W"] @ z + P["B"]
        else:  # W1: T x d x h, b1: T x h, w2: T x h, b2: T
            a = np.maximum(np.einsum("d,tdh->th", z, P["W1"]) + P["b1"], 0.0)
            s = (a * P["w2"]).sum(1) + P["b2"]
        return 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))


class InstrumentDetector:
    """Live vocals / drums / bass / melodic-instrument detection: the student model trained by
    tools/train_vocal_model.py (taught by Demucs htdemucs_ft on the user's songs + their instrumental
    versions). v2: every 0.25 s (GenreDetector), "full" head = YAMNet embedding + vocal_dsp of the last
    1 s, or "lite" head = log YAMNet scores + vocal_dsp; smoothed (EMA tau from the file, 1 s) with
    hysteresis (on > 0.6, off < 0.4). v1 files (YAMNet scores + analyzer DSP, every 0.5 s) still
    load. Plus instrument families straight from YAMNet's classes (EMA ~2 s)."""

    def __init__(self, path=STUDENT_PATH, names=None):
        d = np.load(path)
        self.version = int(d["version"])
        if self.version not in (1, 2):
            raise ValueError(f"{path} is version {self.version}, expected 1 or {STUDENT_VERSION}: retrain")
        self.targets = [str(x) for x in d["targets"]]
        self.n_songs = int(d["n_songs"])
        self.heads = {}
        self.tau, self.th_on, self.th_off = 1.0, {k: 0.6 for k in self.targets}, {k: 0.4 for k in self.targets}
        if self.version == 1:  # logistic regressions on log YAMNet scores + the analyzer's DSP features
            self.W, self.B, self.mu, self.sd = d["W"], d["B"], d["mu"], d["sd"]
        else:  # "full": YAMNet embedding + vocal_dsp, "lite": log YAMNet scores + vocal_dsp
            self.heads = {h: _Head(d, h) for h in ("full", "lite") if f"{h}_kind" in d.files}
            self.tau = float(d["tau"])
            self.th_on = dict(zip(self.targets, map(float, d["th_on"])))
            self.th_off = dict(zip(self.targets, map(float, d["th_off"])))
        self.needs_embedding = "full" in self.heads
        self.head_used = None
        self.p = {k: 0.0 for k in self.targets}
        self.s = {k: 0.0 for k in self.targets}
        self.on = {k: False for k in self.targets}
        self.fam_idx = {f: [i for i, n in enumerate(names or []) if n in cls] for f, cls in FAMILIES.items()}
        self.fam = {f: 0.0 for f in FAMILIES}
        self.lock = threading.Lock()
        self.n = 0

    def predict(self, yam, dsp=None, emb=None, x=None, sr=None):
        """Raw (unsmoothed) probabilities per target, or None if the needed inputs are missing.
        v1: yam + dsp (analyzer). v2: x / sr (the last ~1 s of audio) + emb (YAMNet embedding, for
        the "full" head) or yam (YAMNet scores, "lite" head, used when no embedding)."""
        if self.version == 1:
            if dsp is None:
                return None
            self.head_used = "v1"
            return 1.0 / (1.0 + np.exp(-(self.W @ ((student_vector(yam, dsp) - self.mu) / self.sd) + self.B)))
        if x is None or sr is None:
            return None
        vd = vocal_dsp(x, sr)
        if emb is not None and "full" in self.heads:
            self.head_used = "full"
            return self.heads["full"](np.concatenate([emb, vd]))
        if "lite" in self.heads and yam is not None:
            self.head_used = "lite"
            return self.heads["lite"](np.concatenate([np.log(np.asarray(yam, np.float64) + 1e-3), vd]))
        return None

    def update(self, yam, dsp, period=0.5, emb=None, x=None, sr=None):
        """One student step (every `period` s): raw probabilities -> EMA (~tau s) -> hysteresis per
        target; instrument families straight from YAMNet's scores (EMA ~2 s)."""
        p = self.predict(yam, dsp, emb, x, sr)
        if p is None:
            return
        a = 1.0 - math.exp(-period / self.tau) if self.n else 1.0
        af = 1.0 - math.exp(-period / 2.0) if self.n else 1.0
        with self.lock:
            self.n += 1
            for i, k in enumerate(self.targets):
                self.p[k] = float(p[i])
                self.s[k] += (self.p[k] - self.s[k]) * a
                self.on[k] = self.s[k] > (self.th_off[k] if self.on[k] else self.th_on[k])
            for f, idx in self.fam_idx.items():
                v = float(max(yam[i] for i in idx)) if idx and yam is not None else 0.0
                self.fam[f] += (v - self.fam[f]) * af

    def snapshot(self):
        with self.lock:
            fams = [f for f, v in self.fam.items() if v > 0.08]
            return {"on": dict(self.on), "p": dict(self.s), "fam": dict(self.fam), "families": fams}

    def tags(self):
        s = self.snapshot()
        t = [k.upper() for k in ("drums", "bass") if s["on"][k]]
        t += [f for f in s["families"] if f not in ("CHOIR",)]
        if s["on"]["other"] and not s["families"]:
            t.append("MELODIC")
        if s["on"]["vocals"]:
            t.append("VOX")
        return " ".join(t) or "-"



def pitch_clarity(x, sr):
    """Fraction of 40 ms sub-frames whose normalized autocorrelation has a pitch peak (70-500 Hz)
    > 0.45 (one clear pitch: a solo voice)."""
    n, lo, hi, v = int(0.04 * sr), int(sr / 500), int(sr / 70), []
    for i in range(0, len(x) - n, n):
        s = x[i:i + n] - x[i:i + n].mean()
        e = float((s * s).sum())
        if e < 1e-8:
            v.append(0.0)
            continue
        f = np.fft.rfft(s, 2 * n)
        ac = np.fft.irfft(np.abs(f) ** 2)[:n] / e
        v.append(float(ac[lo:hi].max() > 0.45))
    return float(np.mean(v)) if v else 0.0


def peak_share(x, sr):
    """Polyphony-tolerant voicing: mean share of the 100-4000 Hz energy held by spectral peaks (local
    maxima > 4x the level 3 bins away), 1024-pt frames, hop 512 (choirs: many partials, high)."""
    n, hop = 1024, 512
    if len(x) < n:
        return 0.0
    fr = np.lib.stride_tricks.sliding_window_view(np.asarray(x, np.float64), n)[::hop] * np.hanning(n)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    S = np.abs(np.fft.rfft(fr, axis=1))[:, (f >= 100) & (f < 4000)] ** 2 + 1e-12
    pk = (S >= np.maximum(np.roll(S, 1, 1), np.roll(S, -1, 1))) & (S > 2.0 * (np.roll(S, 3, 1) + np.roll(S, -3, 1)))
    return float(np.mean((S * pk).sum(1) / S.sum(1)))


VOX_STEM_CLASSES = ("Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue", "Shout", "Yell",
                    "Chant", "Mantra", "Singing", "Choir", "Yodeling", "Child singing", "Synthetic singing",
                    "Rapping", "Humming", "Vocal music", "A capella", "Whispering", "Children shouting",
                    "Screaming")


class VocalSeparator:
    """Real vocal detection with a source-separation model: Demucs htdemucs (pretrained, Meta) on
    Apple-Silicon GPU (MPS) or CPU. Every `period` s the last `window` s of audio are separated (7 s
    chunks: htdemucs is trained on 7.8 s); over [-2 s, -0.5 s] (the newest edge of a chunk is
    unreliable):
      ratio_db  vocal stem RMS relative to the mix;
      vox       YAMNet's vocal classes (Singing, Choir, Speech, ...) ON THE SEPARATED VOCAL STEM (two
                windows, ~3 ms; far more reliable than on the mix), averaged over the last 2 chunks;
      tonal     the stem is voiced: one clear pitch (pitch_clarity) or many partials (peak_share:
                choirs).
    Gate (v2, same rule as the teacher of tools/train_vocal_model.py): opens on ratio > -15 dB with
    vox >= 0.05 (and tonal or vox >= 0.15), or ratio > -26 dB with vox >= 0.15 and tonal (choirs /
    backing vocals under an orchestra); stays open while ratio > -20 dB with vox >= 0.03, or ratio >
    -30 dB with vox >= 0.08. Without YAMNet (use_yamnet=False or no mediapipe): the v1 gate, open
    above -15 dB, closed under -20 dB.
    Mono input loses wide-stereo choirs (htdemucs separates them only from stereo: measured on
    Malevolent Shrine, -60 dB mono vs -13 dB stereo): if the analyzer offers
    recent_audio_stereo(seconds) -> (2, n) it is used.
    Measured: ~0.22-0.45 s per chunk on MPS (Apple GPU), ~1 s on CPU; gate delay ~1.5-2.5 s.
    Needs: uv ... --with demucs --with torchaudio (the model, ~80 MB, downloads on first use)."""
    RULE = dict(hi=-15.0, lo=-26.0, vs=0.05, vw=0.15, off_hi=-20.0, off_lo=-30.0, off_vs=0.03, off_vw=0.08,
                clarity=0.25, peak=0.35)

    def __init__(self, period=1.0, window=7.0, device=None, use_yamnet=True):
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        self.torch, self.ta, self.apply = torch, torchaudio, apply_model
        torch.set_num_threads(2)  # leave CPU to the audio, analysis and 50 Hz control threads
        self.dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.m = get_model("htdemucs")
        self.m.eval()
        self.m.to(self.dev)
        self.vi = self.m.sources.index("vocals")
        self.period, self.window = period, window
        self.clf, self.vox_idx = None, []
        if use_yamnet:
            try:  # own MediaPipe instance (this thread), not the genre detector's
                from .genre import make_yamnet, yamnet_names
                self.clf = make_yamnet()
                names = yamnet_names(self.clf)
                self.vox_idx = [names.index(c) for c in VOX_STEM_CLASSES if c in names]
            except Exception as e:
                print(f"vocals: YAMNet on the vocal stem disabled ({type(e).__name__}: {e}); ratio-only gate")
                self.clf = None
        self.lock = threading.Lock()
        self.ratio_db, self.gate, self.t, self.vox, self.tonal = -90.0, False, 0.0, 0.0, False
        self._vox_hist = []
        self.costs = []
        self.timeline = []  # (t, ratio_db, gate, vox)

    def separate(self, x, sr, t):
        """x: mono (n,) or stereo (2, n) float audio at sr ending at stream time t."""
        torch = self.torch
        t0 = time.perf_counter()
        x = np.ascontiguousarray(x, np.float32)
        y = self.ta.functional.resample(torch.from_numpy(x), sr, self.m.samplerate)
        wav = (y if y.ndim == 2 else torch.stack([y, y]))[None]
        with torch.no_grad():
            out = self.apply(self.m, wav.to(self.dev), device=self.dev, split=False, progress=False)
        v = out[0, self.vi].mean(0).cpu().numpy()
        mix = wav[0].mean(0).numpy()
        msr = self.m.samplerate
        a, b = len(v) - int(2.0 * msr), len(v) - int(0.5 * msr)
        rv = float(np.sqrt(np.mean(v[a:b] ** 2)))
        rm = float(np.sqrt(np.mean(mix[a:b] ** 2))) + 1e-9
        db = 20.0 * math.log10(rv / rm + 1e-6)
        vox, tonal = 0.0, False
        if self.clf is not None and db > -40.0:
            from .genre import yamnet_window_scores
            seg = v[a - int(0.5 * msr):b]  # [-2.5, -0.5] s: two YAMNet windows ending at -1.025 / -0.5 s
            s1 = yamnet_window_scores(self.clf, seg[:len(seg) - int(0.525 * msr)], msr)
            s2 = yamnet_window_scores(self.clf, seg, msr)
            vox = float((s1[self.vox_idx].sum() + s2[self.vox_idx].sum()) / 2)
            v22 = v[a:b][::2]
            tonal = (pitch_clarity(v22[::2], msr // 4) > self.RULE["clarity"]
                     or peak_share(v22, msr // 2) > self.RULE["peak"])
        with self.lock:
            self._vox_hist = (self._vox_hist + [vox])[-2:]
            vs = float(np.mean(self._vox_hist))
            self.ratio_db, self.t, self.vox, self.tonal = db, t, vs, tonal
            if self.clf is None:
                self.gate = db > (-20.0 if self.gate else -15.0)
            else:
                R = self.RULE
                if not self.gate:
                    self.gate = bool((db > R["hi"] and vs >= R["vs"] and (tonal or vs >= R["vw"]))
                                     or (db > R["lo"] and vs >= R["vw"] and tonal))
                else:
                    self.gate = bool((db > R["off_hi"] and vs >= R["off_vs"]) or (db > R["off_lo"] and vs >= R["off_vw"]))
            self.costs.append(time.perf_counter() - t0)
            self.timeline.append((t, db, self.gate, vs))

    def snapshot(self):
        with self.lock:
            return {"gate": self.gate, "ratio_db": self.ratio_db, "vox": self.vox, "tonal": self.tonal}

    def run(self, an, stop):
        nxt = time.monotonic()
        stereo = getattr(an, "recent_audio_stereo", None)
        while not stop.is_set():
            x = stereo(self.window) if stereo is not None else an.recent_audio(self.window)
            if x is not None and an.snapshot().section != "silence":
                try:
                    self.separate(x, an.sr, an.samples / an.sr)
                except Exception as e:
                    print(f"vocals: separation failed ({e})")
            nxt += self.period
            stop.wait(max(0.0, nxt - time.monotonic()))


def open_vocals(args):
    if not args.vocal_model:
        return None
    try:
        v = VocalSeparator()
        print(f"vocal detection: Demucs htdemucs on {v.dev} every 1 s")
        return v
    except Exception as e:
        print(f"vocal separation disabled ({type(e).__name__}: {e}); run with uv ... --with demucs "
              "--with torchaudio. Falling back to YAMNet + DSP vocal cues.")
        return None

def vocal_gate(p_student, demucs_gate, was_on, on=0.6, off=0.4, boost=0.25):
    """Fusion of the two vocal detectors (leave-one-song-out benchmark: see the report of
    tools/train_vocal_model.py). p_student: smoothed probability of the trained student (None if not
    loaded), demucs_gate: the live Demucs gate (None if not loaded).
    v2: the student decides (it reacts in ~1 s and, unlike the mono 7-s Demucs chunks, hears wide-
    stereo choirs); an open Demucs gate adds `boost` to its probability (vocals the student is unsure
    about, e.g. under a loud mix). Hysteresis on > on, off < off. (v1 was 0.6 * student + 0.4 * gate:
    in effect Demucs AND student, which missed every choir Demucs could not separate.)"""
    if p_student is None:
        return bool(demucs_gate)
    v = p_student + (boost if demucs_gate else 0.0)
    return v > (off if was_on else on)


class VocalFusion:
    """Per control tick: vocals on/off (hysteresis), mouth opening 0..1 and kind of voice, from the
    student (if loaded), the Demucs gate (if loaded), YAMNet's vocal classes and the fast DSP
    envelope of the harmonic voice-band energy."""

    def __init__(self):
        self.vox_on = self.song_vox = False
        self.vox_act = 0.0
        self.vox_text = "VOX -"

    def update(self, st, g, dt, student=None, separator=None):
        """st: BeatState, g: GenreDetector.snapshot() or None, student: InstrumentDetector or None,
        separator: VocalSeparator or None. Returns {"on", "mouth", "kind"}.
        YAMNet's vocal classes (~1 s, weak in full mixes: 0.005-0.05) say whether the song has a voice
        now; the fast DSP envelope of the harmonic voice-band energy (~50 ms) says when a syllable /
        line is sung."""
        self.vox_act += (st.vox_env - self.vox_act) * _ema(dt, 0.15)
        inst = student.snapshot() if student else None
        if inst is not None or separator is not None:
            # trained student (fast, every 0.25 s) and/or Demucs gate (slow, ~2 s): see vocal_gate()
            dg = separator.snapshot()["gate"] if separator is not None else None
            sp = inst["p"]["vocals"] if inst is not None else None
            th = {}
            if student is not None and hasattr(student, "th_on"):
                th = {"on": student.th_on.get("vocals", 0.6), "off": student.th_off.get("vocals", 0.4)}
            self.song_vox = vocal_gate(sp, dg, self.song_vox, **th)
            self.vox_on = st.section != "silence" and self.song_vox
            kind = ((g["vox_kind"] if g is not None and g["vox_kind"] != "none" else "sung")
                    if self.vox_on else "none")
            self.vox_text = f"VOX {kind} {st.vox_env:.1f}" if self.vox_on else "VOX -"
            return {"on": self.vox_on, "mouth": st.vox_env if self.vox_on else 0.0, "kind": kind}
        if g is not None:
            p = g["vox_p"]
            self.song_vox = p > 0.006 if not self.song_vox else p > 0.003
        else:
            self.song_vox = True  # no classifier: trust the DSP envelope alone
        active = self.vox_act > (0.35 if self.vox_on else 0.5)
        self.vox_on = st.section != "silence" and self.song_vox and active
        kind = (g["vox_kind"] if g is not None else "sung") if self.vox_on else "none"
        self.vox_text = f"VOX {kind} {st.vox_env:.1f}" if self.vox_on else "VOX -"
        return {"on": self.vox_on, "mouth": st.vox_env if self.vox_on else 0.0, "kind": kind}
