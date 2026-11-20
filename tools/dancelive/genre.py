"""Genre -> dance style: YAMNet (521 AudioSet classes, MediaPipe Tasks, CPU) wrappers (scores, one-window
scores, and the 1024-d embedding through a patched copy of the model), the genre / instrument ->
style tables, and GenreDetector (EMA, style hysteresis, vocal-kind classes; it also feeds the vocal
/ instrument student: every 0.25 s with a v2 student, the genre part every 0.5 s)."""
import math
import threading
import time
from pathlib import Path

import numpy as np

from .vocals import STUDENT_PATH, InstrumentDetector

# ================================================================ YAMNet
YAMNET_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite"
YAMNET_PATH = Path.home() / ".cache/so101/yamnet.tflite"


def make_yamnet(model=None):
    """YAMNet (521 AudioSet classes) through MediaPipe on CPU; downloads the model once."""
    from mediapipe.tasks.python import BaseOptions, audio
    model = Path(model or YAMNET_PATH)
    if not model.exists():
        import urllib.request
        model.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YAMNET_URL, model)
    return audio.AudioClassifier.create_from_options(audio.AudioClassifierOptions(
        base_options=BaseOptions(model_asset_path=str(model)), max_results=-1,
        running_mode=audio.RunningMode.AUDIO_CLIPS))


def yamnet_scores(clf, x, sr):
    """521 class scores (index order) of audio x (any rate; FFT-resampled to 16 kHz), averaged over
    YAMNet's ~1 s windows."""
    from mediapipe.tasks.python.components.containers import audio_data
    n16 = int(round(len(x) * 16000 / sr))
    y = np.fft.irfft(np.fft.rfft(x)[:n16 // 2 + 1], n16).astype(np.float32) * (n16 / len(x))
    res = clf.classify(audio_data.AudioData.create_from_array(y, 16000))
    v = np.zeros(521, np.float32)
    for r in res:
        for c in r.classifications[0].categories:
            v[c.index] += c.score / len(res)
    return v


YAMNET_EMBED_PATH = Path.home() / ".cache/so101/yamnet_embed.tflite"
YAMNET_WIN = 15600  # samples at 16 kHz in one YAMNet window (0.975 s)


def _resample16(x, sr):
    """FFT resampling to 16 kHz (as yamnet_scores)."""
    n16 = int(round(len(x) * 16000 / sr))
    return np.fft.irfft(np.fft.rfft(x)[:n16 // 2 + 1], n16).astype(np.float32) * (n16 / len(x))


def make_embed_model(src=YAMNET_PATH, dst=YAMNET_EMBED_PATH):
    """A copy of yamnet.tflite whose output is YAMNet's 1024-d embedding (layer28 global average,
    dequantized) instead of its 521 class scores, so MediaPipe's AudioClassifier returns the
    embedding as 1024 'categories'. Pure-python flatbuffer patch (two 4-byte edits: the final
    DEQUANTIZE reads the embedding tensor, the output shape [1,521] -> [1,1024]) + a 1024-line label
    file in the model's zip. No TensorFlow needed."""
    import io
    import struct
    import zipfile
    b = bytearray(Path(src).read_bytes())
    u32 = lambda p: struct.unpack_from("<I", b, p)[0]  # noqa: E731

    def field(t, i):
        vt = t - struct.unpack_from("<i", b, t)[0]
        o = 4 + 2 * i
        if o >= struct.unpack_from("<H", b, vt)[0]:
            return None
        off = struct.unpack_from("<H", b, vt + o)[0]
        return t + off if off else None

    def vec(t, i):
        p = field(t, i)
        p += u32(p)
        return p + 4, u32(p)

    def tables(t, i):
        p, n = vec(t, i)
        return [p + 4 * k + u32(p + 4 * k) for k in range(n)]

    def tname(t):
        p, n = vec(t, 3)
        return bytes(b[p:p + n]).decode()

    sg = tables(u32(0), 2)[0]  # Model.subgraphs[0]
    tensors = tables(sg, 0)
    emb = [k for k, t in enumerate(tensors) if tname(t).endswith("layer28/reduce_mean")]
    po, no = vec(sg, 2)
    if len(emb) != 1 or no != 1:
        raise ValueError("unexpected YAMNet graph")
    out = u32(po)
    patched = 0
    for op in tables(sg, 3):
        pi, ni = vec(op, 1)
        pq, nq = vec(op, 2)
        if out in [u32(pq + 4 * k) for k in range(nq)]:
            if ni != 1:
                raise ValueError("unexpected YAMNet output op")
            struct.pack_into("<I", b, pi, emb[0])
            patched += 1
    ps, ns = vec(tensors[out], 0)
    if patched != 1 or ns != 2 or u32(ps + 4) != 521:
        raise ValueError("unexpected YAMNet output tensor")
    struct.pack_into("<I", b, ps + 4, 1024)
    zi = zipfile.ZipFile(io.BytesIO(bytes(b))).infolist()[0]
    tmp = Path(str(dst) + ".tmp")
    tmp.write_bytes(bytes(b[:zi.header_offset]))
    with zipfile.ZipFile(tmp, "a", compression=zipfile.ZIP_STORED) as z:
        z.writestr(zi.filename, "\n".join(f"e{i}" for i in range(1024)) + "\n")
    tmp.replace(dst)
    return Path(dst)


def make_yamnet_embedder(model=None):
    """YAMNet's 1024-d embedding through MediaPipe (CPU, ~1.5 ms per 0.975 s window): an
    AudioClassifier on the patched model of make_embed_model() (created once next to yamnet.tflite)."""
    from mediapipe.tasks.python import BaseOptions, audio
    model = Path(model or YAMNET_EMBED_PATH)
    if not model.exists():
        if not YAMNET_PATH.exists():
            make_yamnet().close()  # downloads yamnet.tflite
        make_embed_model(YAMNET_PATH, model)
    return audio.AudioClassifier.create_from_options(audio.AudioClassifierOptions(
        base_options=BaseOptions(model_asset_path=str(model)), max_results=-1,
        running_mode=audio.RunningMode.AUDIO_CLIPS))


def yamnet_embedding(emb, x, sr):
    """1024-d YAMNet embedding of the LAST 0.975 s of audio x (any rate, >= 0.975 s)."""
    from mediapipe.tasks.python.components.containers import audio_data
    y = _resample16(x, sr)[-YAMNET_WIN:]
    if len(y) < YAMNET_WIN:
        y = np.pad(y, (YAMNET_WIN - len(y), 0))
    res = emb.classify(audio_data.AudioData.create_from_array(y, 16000))
    v = np.zeros(1024, np.float32)
    for c in res[0].classifications[0].categories:
        v[c.index] = c.score
    return v


def yamnet_window_scores(clf, x, sr):
    """521 class scores of exactly one YAMNet window: the LAST 0.975 s of x. (yamnet_scores() on 1 s
    averages a real window with a mostly zero-padded one, i.e. about halves the scores; the genre
    tables are tuned on that, this one is for the vocal models.)"""
    from mediapipe.tasks.python.components.containers import audio_data
    y = _resample16(x, sr)[-YAMNET_WIN:]
    if len(y) < YAMNET_WIN:
        y = np.pad(y, (YAMNET_WIN - len(y), 0))
    res = clf.classify(audio_data.AudioData.create_from_array(y, 16000))
    v = np.zeros(521, np.float32)
    for c in res[0].classifications[0].categories:
        v[c.index] = c.score
    return v


def yamnet_names(clf):
    from mediapipe.tasks.python.components.containers import audio_data
    res = clf.classify(audio_data.AudioData.create_from_array(np.zeros(16000, np.float32), 16000))
    names = [""] * 521
    for c in res[0].classifications[0].categories:
        names[c.index] = c.category_name
    return names


# ================================================================ genre -> dance style
GENRE_STYLE = {  # YAMNet (AudioSet) genre classes -> dance style
    "Electronic music": "electronic", "Techno": "electronic", "House music": "electronic",
    "Trance music": "electronic", "Dubstep": "electronic", "Drum and bass": "electronic",
    "Electronica": "electronic", "Electronic dance music": "electronic", "Disco": "electronic",
    "Hip hop music": "hiphop", "Rhythm and blues": "hiphop", "Funk": "hiphop", "Soul music": "hiphop",
    "Beatboxing": "hiphop", "Afrobeat": "hiphop",
    "Rock music": "rock", "Heavy metal": "rock", "Punk rock": "rock", "Grunge": "rock",
    "Progressive rock": "rock", "Rock and roll": "rock", "Psychedelic rock": "rock", "Ska": "rock",
    "Pop music": "pop", "Music of Asia": "pop", "Video game music": "pop", "Soundtrack music": "pop",
    "Theme music": "pop", "Dance music": "pop", "Independent music": "pop", "Music of Bollywood": "pop",
    "Christmas music": "pop", "Music for children": "pop",
    "Music of Latin America": "latin", "Salsa music": "latin", "Flamenco": "latin", "Reggae": "latin",
    "Swing music": "calm",
    "Classical music": "calm", "Jazz": "calm", "Ambient music": "calm", "Lullaby": "calm",
    "New-age music": "calm", "Opera": "calm", "Blues": "calm", "Country": "calm", "Folk music": "calm",
    "Bluegrass": "calm", "Gospel music": "calm", "Christian music": "calm", "Traditional music": "calm",
    "Middle Eastern music": "calm", "Carnatic music": "calm", "A capella": "calm", "Vocal music": "calm",
}
# YAMNet's genre scores are small (typically 0.01-0.06 next to "Music" 0.6) while its instrument
# scores are confident: instruments are weaker evidence (x INSTRUMENT_WEIGHT) for a style.
INSTRUMENT_STYLE = {
    "Drum machine": "electronic", "Synthesizer": "electronic", "Sampler": "electronic",
    "Electronic organ": "electronic",
    "Scratching (performance technique)": "hiphop", "Rapping": "hiphop",
    "Electric guitar": "rock", "Drum kit": "rock", "Bass guitar": "rock", "Cymbal": "rock",
    "Maraca": "latin", "Tambourine": "latin", "Marimba, xylophone": "latin", "Steelpan": "latin",
    "Accordion": "latin", "Trumpet": "latin",
    "Piano": "calm", "Acoustic guitar": "calm", "Orchestra": "calm", "String section": "calm",
    "Violin, fiddle": "calm", "Cello": "calm", "Harp": "calm", "Flute": "calm", "Choir": "calm",
    "Bowed string instrument": "calm", "Saxophone": "calm",
}
INSTRUMENT_WEIGHT = 0.25
VOX_CLASSES = {  # YAMNet vocal classes -> kind of voice
    "Singing": "sung", "Male singing": "sung", "Female singing": "sung", "Child singing": "sung",
    "Synthetic singing": "sung", "Humming": "sung", "Rapping": "rap", "Choir": "choir", "Chant": "choir",
    "Vocal music": "sung", "A capella": "choir", "Yodeling": "sung",
}
STYLE_WEIGHT = {"pop": 0.85}  # YAMNet says "Pop music" for almost any song: a specific genre wins ties


class GenreDetector:
    """YAMNet (AudioSet, 521 classes) through MediaPipe Tasks on CPU: every `period` s the last
    `window` s of audio (resampled to 16 kHz) is classified; the genre classes are smoothed with an
    EMA (~tau s) and summed per dance style; the style label only switches when another style leads
    by 30 % for `hold` s. A "not music" flag (Music score low, or Speech above Music) keeps the
    current style. Thread-safe: classify() is called by its own thread (or inline in --fast)."""

    def __init__(self, model=YAMNET_PATH, tau=15.0, period=None, window=1.0, hold=5.0, student=True,
                 genre_period=0.5, student_period=0.25):
        """period: how often classify() should be called (the thread / --fast loop use it): the
        student's period (0.25 s) with a v2 student, else genre_period. The genre / style part
        updates every genre_period s whatever the call rate."""
        self.clf = make_yamnet(model)
        self.names = yamnet_names(self.clf)
        self.student = None
        self.embedder = None
        if student:
            try:
                self.student = InstrumentDetector(STUDENT_PATH, self.names)
                print(f"vocals / instruments: student model ({STUDENT_PATH.name}, {self.student.n_songs} songs)")
            except Exception as e:
                print(f"vocals / instruments student disabled ({type(e).__name__}: {e}); train it with "
                      "tools/train_vocal_model.py")
        if self.student is not None and getattr(self.student, "needs_embedding", False):
            try:
                self.embedder = make_yamnet_embedder()
            except Exception as e:
                print(f"vocals: YAMNet embedding unavailable ({type(e).__name__}: {e}); student uses its lite head")
        if period is None:
            period = student_period if self.student is not None and self.student.version >= 2 else genre_period
        self.tau, self.period, self.window, self.hold = tau, period, window, hold
        self.genre_period = max(genre_period, period)
        self._t_genre = None
        self.lock = threading.Lock()
        self.ema = {g: 0.0 for g in GENRE_STYLE}
        self.ema_inst = {g: 0.0 for g in INSTRUMENT_STYLE}
        self.vox = {"sung": 0.0, "rap": 0.0, "choir": 0.0}  # EMA ~3 s of the vocal classes per kind
        self.music = self.speech = 0.0
        self.style, self.cand, self.cand_since = None, None, 0.0
        self.n = 0
        self.min_n = 2  # classifications before the first decision (fast start, refined later)
        self.early_n = 20  # during the first ~20 s a lead switches the style faster
        self.no_pulse = 0.0
        self.costs = []  # s per classification
        self.timeline = []  # (t, style, top-3)

    def classify(self, x, sr, t, pulse=1.0, dsp=None):
        """x: audio at sr; t: stream time; pulse: 0..1 how periodic the rhythm is (analyzer):
        music without a steady pulse is evidence for the calm style; dsp: the analyzer's feature
        vector for the vocal / instrument student."""
        t0 = time.perf_counter()
        yam = yamnet_scores(self.clf, x, sr)
        if self.student is not None:
            emb = None
            if self.embedder is not None:
                try:
                    emb = yamnet_embedding(self.embedder, x, sr)
                except Exception as e:  # keep dancing on the lite head
                    print(f"vocals: YAMNet embedding failed ({e}); lite head from now on")
                    self.embedder = None
            self.student.update(yam, dsp, self.period, emb=emb, x=x, sr=sr)
        cost = time.perf_counter() - t0
        if self._t_genre is not None and t - self._t_genre < self.genre_period - 1e-6:
            with self.lock:
                self.costs.append(cost)
            return  # student-only step: the genre part runs every genre_period s
        self._t_genre = t
        scores = {n: float(v) for n, v in zip(self.names, yam)}
        # running mean for the first samples (a quick first estimate), then an EMA (~tau s)
        gp = self.genre_period
        a = max(1.0 - math.exp(-gp / self.tau), 1.0 / (self.n + 1))
        with self.lock:
            self.n += 1
            self.costs.append(cost)
            self.music += (scores.get("Music", 0.0) - self.music) * a
            self.speech += (scores.get("Speech", 0.0) - self.speech) * a
            for g in self.ema:
                self.ema[g] += (scores.get(g, 0.0) - self.ema[g]) * a
            for g in self.ema_inst:
                self.ema_inst[g] += (scores.get(g, 0.0) - self.ema_inst[g]) * a
            ap = 1.0 - math.exp(-gp / 4.0) if self.n > 1 else 1.0  # the pulse cue reacts fast
            self.no_pulse += ((1.0 - min(max(pulse, 0.0), 1.0)) - self.no_pulse) * ap
            av = max(1.0 - math.exp(-gp / 3.0), 1.0 / (self.n + 1))
            per = {k: 0.0 for k in self.vox}
            for c, kind in VOX_CLASSES.items():
                per[kind] += scores.get(c, 0.0)
            for kind in self.vox:
                self.vox[kind] += (per[kind] - self.vox[kind]) * av
            st = self._style_scores()
            best = max(st, key=st.get)
            if self.is_music() and st[best] > 0.02 and self.n >= self.min_n:
                if self.style is None:
                    self.style = best
                elif best != self.style and st[best] > (1.15 if self.n < self.early_n else 1.3) * st[self.style]:
                    if self.cand != best:
                        self.cand, self.cand_since = best, t
                    elif t - self.cand_since >= (2.0 if self.n < self.early_n else self.hold):
                        self.style, self.cand = best, None
                else:
                    self.cand = None
            self.timeline.append((t, self.style, self.top(3)))

    def is_music(self):
        return self.music > 0.15 and self.music > self.speech

    def _style_scores(self):
        st = {}
        for g, v in self.ema.items():
            k = GENRE_STYLE[g]
            st[k] = st.get(k, 0.0) + v * STYLE_WEIGHT.get(k, 1.0)
        for g, v in self.ema_inst.items():
            k = INSTRUMENT_STYLE[g]
            st[k] = st.get(k, 0.0) + v * INSTRUMENT_WEIGHT
        st["calm"] = st.get("calm", 0.0) + 0.05 * self.no_pulse
        st["hiphop"] = st.get("hiphop", 0.0) + self.vox["rap"]  # rapping -> hip-hop
        if self.student is not None and self.student.n:  # instruments: guitar+drums rock, synth+drums
            d = self.student.s["drums"]                    # electronic, piano/strings w/o drums calm
            f = self.student.fam
            st["rock"] = st.get("rock", 0.0) + 0.04 * min(f["GTR"] / 0.2, 1.0) * d
            st["electronic"] = st.get("electronic", 0.0) + 0.04 * min(f["SYNTH"] / 0.2, 1.0) * d
            st["calm"] = st.get("calm", 0.0) + 0.04 * min((f["PIANO"] + f["STRINGS"]) / 0.2, 1.0) * (1.0 - d)
        return st

    def top(self, k=3):
        """Strongest genre classes (instruments weighted as in the style score)."""
        both = {**self.ema, **{g: v * INSTRUMENT_WEIGHT for g, v in self.ema_inst.items()}}
        return sorted(both.items(), key=lambda kv: -kv[1])[:k]

    def snapshot(self):
        with self.lock:
            styles = self._style_scores()
            vox_p = sum(self.vox.values())
            return {"style": self.style, "top": self.top(3), "music": self.music, "speech": self.speech,
                    "is_music": self.is_music(), "styles": styles,
                    # calm evidence close to the current style: lets a calm section soften the dance
                    "prefers_calm": bool(self.style and styles.get("calm", 0.0) >= 0.6 * styles.get(self.style, 1.0)),
                    "vox_p": vox_p, "vox_kind": max(self.vox, key=self.vox.get) if vox_p > 0 else "none"}

    def text(self):
        g = self.snapshot()
        top = ", ".join(f"{n.replace(' music', '')} {v:.2f}" for n, v in g["top"])
        return f"genre: {g['style'] or '?'} ({top})" + ("" if g["is_music"] else " [not music]")

    def run(self, an, stop):
        """Thread body: classify the analyzer's latest audio every `period` s."""
        nxt = time.monotonic()
        while not stop.is_set():
            x = an.recent_audio(self.window)
            if x is not None and an.snapshot().section != "silence":
                try:
                    self.classify(x, an.sr, an.samples / an.sr, an.rhythm(), an.feature_vector())
                except Exception as e:  # never take the dance down
                    print(f"genre: classification failed ({e})")
            nxt += self.period
            stop.wait(max(0.0, nxt - time.monotonic()))


def open_genre(args):
    if not args.genre:
        return None
    try:
        g = GenreDetector()
        print(f"genre detection: YAMNet ({YAMNET_PATH.name}, MediaPipe) every {g.genre_period:.2f} s"
              + (f", vocal student every {g.period:.2f} s ({'embedding' if g.embedder is not None else 'lite'} head)"
                 if g.student is not None else ""))
        return g
    except Exception as e:
        print(f"genre detection disabled ({type(e).__name__}: {e}); run with uv ... --with mediapipe")
        return None
