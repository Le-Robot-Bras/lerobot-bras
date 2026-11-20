#!/usr/bin/env python3
"""train_vocal_model.py - teaches tools/dance_live.py to hear vocals and instruments live.

    uv run --no-project --python 3.12 --with demucs --with torchaudio --with mediapipe --with numpy \
        tools/train_vocal_model.py --songs /path/to/your/music          # all songs in that folder
    ... tools/train_vocal_model.py --songs ~/Music/a --songs ~/Music/b     # several folders
    ... tools/train_vocal_model.py --songs /path/to/your/music --relabel   # recompute the teacher labels

Re-runnable: every song is cached in ~/.cache/so101/labels/ under its CONTENT hash (+ the label /
feature version), so a new run only processes new or changed songs.

Teacher (offline, slow, accurate; label version 2): Demucs htdemucs_ft (the fine-tuned bag of 4
htdemucs, ~4x slower, cleaner vocal stem; --teacher htdemucs for speed) separates each full song
into vocals / drums / bass / other. Per 0.5 s frame:
  drums / bass > -20 dB, other > -15 dB relative to the mix (as v1);
  vocals: the vocal stem must be audible (> -35 dB under the song's loud level) and either
    strong:  > -15 dB vs the mix, voiced, and YAMNet on the VOCAL STEM not clearly non-vocal
             (vocal classes >= 0.05), or
    weak:    > -26 dB vs the mix (choirs / backing vocals under an orchestra) with YAMNet on the
             vocal stem clearly vocal (Singing / Choir / Chant / Speech / ... >= 0.15) and voiced;
  "voiced" = pitched autocorrelation in >= 25 % of 40 ms sub-frames (one clear pitch: solo voice) OR
  spectral-peak energy share >= 0.35 (tonal partials of any number of simultaneous pitches: choirs)
  OR YAMNet vocal classes >= 0.15 (rap / speech). YAMNet runs on the separated stem, where it is far
  more reliable than on the mix; its vocal evidence is averaged over 2.5 s; a 3-frame majority
  removes single-frame flips.

Student (what runs live, causal, cheap; feature version 2): every 0.25 s, the last 1 s of audio ->
  "full" head: YAMNet's 1024-d embedding (MediaPipe on a patched yamnet.tflite, ~1.5 ms) + vocal_dsp
              (77 spectral-shape / voice cues computed from the raw audio, ~5 ms);
  "lite" head: log YAMNet class scores (as the genre detector computes them) + vocal_dsp, used live
              when the embedding is not available.
Per target (vocals, drums, bass, other) a logistic regression (L2) or a small MLP (--model mlp),
numpy only, class-balanced PER SONG (each song's positives and negatives weigh the same, so
vocal-sparse songs count). Evaluated by leave-one-song-out cross-validation with the live smoothing
(EMA tau, hysteresis on / off); then trained on every song and saved to
~/.cache/so101/vocal_model.npz (the previous one is kept as vocal_model.prev.npz), which
dance_live.py loads automatically.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dance_live as dl  # noqa: E402

CACHE = Path.home() / ".cache/so101"
FR = 0.5  # s per teacher frame
SP = 0.25  # s per student step (live update period)
TARGETS = dl.STUDENT_TARGETS  # ("vocals", "drums", "bass", "other")
THRESH = {"vocals": -15.0, "drums": -20.0, "bass": -20.0, "other": -15.0}
LABEL_VERSION = 2
FEAT_VERSION = 2
VOCAL_RULE = dict(hi=-15.0, lo=-26.0, v_strong=0.05, v_weak=0.15, audible=-35.0, clarity=0.25, peak=0.35)
VOX_CLASSES = dl.VOX_STEM_CLASSES  # YAMNet classes counted as voice on the separated vocal stem
# live smoothing of the student (EMA time constant, hysteresis); see the report of this script
TAU = 1.0
TH_ON = {"vocals": 0.6, "drums": 0.6, "bass": 0.6, "other": 0.6}
TH_OFF = {"vocals": 0.4, "drums": 0.4, "bass": 0.4, "other": 0.4}


def song_key(path):
    """Cache key: file name stem + hash of the file CONTENT (renamed copies share it)."""
    p = Path(path)
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return p.stem[:40].replace("/", "_") + "-" + h.hexdigest()[:10]


def decode(path, sr, channels):
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "f32le",
                          "-ac", str(channels), "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, channels).T.copy()


clarity = dl.pitch_clarity  # one clear pitch (solo voice)
peak_share = dl.peak_share  # many partials (choirs)


_demucs = {}


def separate(path, model):
    """Full-song Demucs separation -> (sources, stems [S x n] mono at 22.05 kHz, mix mono 22.05 kHz, s)."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    if _demucs.get("name") != model:
        m = get_model(model)
        m.eval()
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        m.to(dev)
        _demucs.update(m=m, dev=dev, name=model)
    m, dev = _demucs["m"], _demucs["dev"]
    y = decode(path, m.samplerate, 2)
    t0 = time.perf_counter()
    with torch.no_grad():
        st = apply_model(m, torch.from_numpy(y)[None], device=dev, split=True, overlap=0.25, progress=False)
    st = st[0].cpu().numpy().mean(1)
    return list(m.sources), st[:, ::2], y.mean(0)[::2], time.perf_counter() - t0  # 44.1 -> 22.05 kHz


def teacher_measure(sources, stems, mix, clf, names, sr=22050):
    """Per 0.5 s frame: level of every stem vs the mix (dB), mix level, vocal-stem voicing (clarity,
    peak share) and YAMNet's vocal-class evidence on the vocal stem (window centred on the frame)."""
    hop = int(FR * sr)
    n = len(mix) // hop

    def rms(a):
        return np.sqrt(np.mean(np.asarray(a[:n * hop], np.float64).reshape(n, hop) ** 2, 1)) + 1e-9

    rm = rms(mix)
    res = {"t": (np.arange(n) + 1) * FR, "mix_db": 20 * np.log10(rm)}
    for k, s in enumerate(sources):
        res[f"ratio_{s}"] = 20 * np.log10(rms(stems[k]) / rm)
    v = np.asarray(stems[sources.index("vocals")], np.float32)
    res["clarity"] = np.array([clarity(v[k * hop:(k + 1) * hop][::2], sr // 2) for k in range(n)])
    res["peak"] = np.array([peak_share(v[k * hop:(k + 1) * hop], sr) for k in range(n)])
    vi = [names.index(c) for c in VOX_CLASSES if c in names]
    yv = np.zeros((n, len(names)), np.float32)
    for k in range(n):
        c = int((k + 0.5) * hop)
        seg = v[max(0, c - int(0.4875 * sr)):c + int(0.4875 * sr)]
        if len(seg) > sr // 4 and np.abs(seg).max() > 1e-5:
            yv[k] = dl.yamnet_window_scores(clf, seg, sr)
    res["yam_stem"] = yv.astype(np.float16)
    res["vox_stem"] = yv[:, vi].sum(1)
    return res


def _smooth(x, k=5):
    return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, "valid")


def label_rule(m, rule=VOCAL_RULE):
    """Teacher labels (bool per 0.5 s frame) from the measurements of teacher_measure()."""
    lab = {s: m[f"ratio_{s}"] > THRESH[s] for s in TARGETS if s != "vocals"}
    r = m["ratio_vocals"]
    V = _smooth(np.asarray(m["vox_stem"], np.float64))
    audible = r + m["mix_db"] - np.percentile(m["mix_db"], 95) > rule["audible"]
    tonal = (m["clarity"] > rule["clarity"]) | (m["peak"] > rule["peak"])
    strong = (r > rule["hi"]) & (tonal | (V >= rule["v_weak"])) & (V >= rule["v_strong"])
    weak = (r > rule["lo"]) & (V >= rule["v_weak"]) & tonal
    y = (audible & (strong | weak)).astype(int)
    z = np.pad(y, 1, mode="edge")
    lab["vocals"] = (z[:-2] + z[1:-1] + z[2:]) >= 2
    return lab


def live_features(y, clf, emb, step=SP, sr=22050):
    """The live student inputs, computed exactly as live: every `step` s from t = 1 s, the last 1 s of
    22.05 kHz mono audio -> YAMNet embedding E, YAMNet scores S (genre detector's call), vocal_dsp D."""
    ts, E, S, D = [], [], [], []
    t = 1.0
    while t * sr <= len(y):
        x = np.asarray(y[int((t - 1.0) * sr):int(t * sr)], np.float32)
        E.append(dl.yamnet_embedding(emb, x, sr))
        S.append(dl.yamnet_scores(clf, x, sr))
        D.append(dl.vocal_dsp(x, sr))
        ts.append(t)
        t += step
    return {"t": np.array(ts), "E": np.array(E, np.float32), "S": np.array(S, np.float32), "D": np.array(D, np.float32)}


def teacher_labels(path, key, clf, emb, names, model="htdemucs_ft", relabel=False):
    """Teacher labels per 0.5 s frame + the hard negatives: live features (every 0.5 s) of the song's
    INSTRUMENTAL version (drums + bass + other stems), i.e. the same music without its voice."""
    lab_f = CACHE / "labels" / f"{key}.teach{LABEL_VERSION}.npz"
    aug_f = CACHE / "labels" / f"{key}.aug{FEAT_VERSION}.npz"
    if relabel or not lab_f.exists() or not aug_f.exists():
        sources, stems, mix, cost = separate(path, model)
        print(f"  teacher {Path(path).name[:40]}: {len(mix) / 22050:.0f} s separated by {model} in {cost:.0f} s",
              flush=True)
        lab_f.parent.mkdir(parents=True, exist_ok=True)
        if relabel or not lab_f.exists():
            m = teacher_measure(sources, stems, mix, clf, names)
            m["teacher"], m["cost"] = model, cost
            np.savez(lab_f, **m)
        if relabel or not aug_f.exists():
            inst = sum(np.asarray(stems[k], np.float32) for k, s in enumerate(sources) if s != "vocals")
            np.savez(aug_f, **live_features(inst, clf, emb, step=2 * SP))
    m = dict(np.load(lab_f))
    lab = label_rule(m)
    return {"t": m["t"], **{f"y_{s}": lab[s] for s in TARGETS}}, dict(np.load(aug_f))


def student_features(path, key, clf, emb):
    out = CACHE / "labels" / f"{key}.feat{FEAT_VERSION}.npz"
    if out.exists():  # bump FEAT_VERSION when the live features change
        return dict(np.load(out))
    res = live_features(decode(path, 22050, 1)[0], clf, emb)
    np.savez(out, **res)
    return res


def head_inputs(fe, head):
    if head == "full":
        return np.concatenate([fe["E"], fe["D"]], 1).astype(np.float64)
    return np.concatenate([np.log(fe["S"].astype(np.float64) + 1e-3), fe["D"]], 1)


# ---------------------------------------------------------------- models (numpy)
def _sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, sw, l2=30.0, iters=300, lr=0.5):
    """Weighted L2 logistic regression by full-batch gradient descent with Nesterov momentum."""
    n, d = X.shape
    w, b, vw, vb = np.zeros(d), 0.0, np.zeros(d), 0.0
    sw = sw / sw.mean()
    for _ in range(iters):
        wn, bn = w + 0.9 * vw, b + 0.9 * vb
        g = (_sig(X @ wn + bn) - y) * sw
        gw = X.T @ g / n + l2 * wn / n
        vw, vb = 0.9 * vw - lr * gw, 0.9 * vb - lr * g.mean()
        w, b = w + vw, b + vb
    return {"W": w, "B": b}


def fit_mlp(X, y, sw, h=32, wd=1e-3, iters=500, lr=3e-3, drop_in=0.2, seed=0):
    """One hidden ReLU layer, Adam, weight decay, input dropout; weighted cross-entropy."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    sw = sw / sw.mean()
    P = {"W1": rng.normal(0, 1 / np.sqrt(d), (d, h)), "b1": np.zeros(h), "w2": rng.normal(0, 1 / np.sqrt(h), h),
         "b2": np.zeros(1)}
    M = {k: np.zeros_like(v) for k, v in P.items()}
    V = {k: np.zeros_like(v) for k, v in P.items()}
    for it in range(1, iters + 1):
        Xd = X * (rng.random(X.shape) > drop_in) / (1 - drop_in) if drop_in else X
        H = Xd @ P["W1"] + P["b1"]
        A = np.maximum(H, 0)
        g = (_sig(A @ P["w2"] + P["b2"][0]) - y) * sw / n
        gA = np.outer(g, P["w2"]) * (H > 0)
        G = {"W1": Xd.T @ gA + wd * P["W1"], "b1": gA.sum(0), "w2": A.T @ g + wd * P["w2"], "b2": np.array([g.sum()])}
        for k in P:
            M[k] = 0.9 * M[k] + 0.1 * G[k]
            V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
            P[k] -= lr * (M[k] / (1 - 0.9 ** it)) / (np.sqrt(V[k] / (1 - 0.999 ** it)) + 1e-8)
    return {"W1": P["W1"], "b1": P["b1"], "w2": P["w2"], "b2": P["b2"][0]}


def predict(kind, m, Z):
    if kind == "lin":
        return _sig(Z @ m["W"] + m["B"])
    return _sig(np.maximum(Z @ m["W1"] + m["b1"], 0) @ m["w2"] + m["b2"])


def song_weights(y, sid):
    """Class-balanced per song: in every song the positives share one unit of weight (scaled down
    when a song has < 20 positive frames) and the negatives one unit; then positives vs negatives
    overall are balanced 50/50."""
    w = np.zeros(len(y))
    for s in np.unique(sid):
        m = sid == s
        pos = m & (y > 0.5)
        neg = m & (y <= 0.5)
        w[pos] = min(1.0, pos.sum() / 20) / max(pos.sum(), 1)
        w[neg] = 1.0 / max(neg.sum(), 1)
    for msk in (y > 0.5, y <= 0.5):
        if w[msk].sum() > 0:
            w[msk] *= 0.5 / w[msk].sum()
    return w * len(y) + 1e-9


def fit(kind, X, y, sid, l2):
    w = song_weights(y, sid)
    if y.min() == y.max():  # degenerate target: constant
        return ({"W": np.zeros(X.shape[1]), "B": 10.0 if y[0] else -10.0} if kind == "lin" else
                {"W1": np.zeros((X.shape[1], 32)), "b1": np.zeros(32), "w2": np.zeros(32), "b2": 10.0 if y[0] else -10.0})
    return fit_logreg(X, y, w, l2) if kind == "lin" else fit_mlp(X, y, w)


def training_set(Xs, As, Ys, songs):
    """The songs' steps + their instrumental versions as vocal negatives (in an instrumental version
    vocals are 0; drums / bass / other keep the song's labels). Every song and every instrumental
    version is its own group for song_weights()."""
    X, Y, sid = [], [], []
    for j in songs:
        X += [Xs[j], As[j]]
        n = len(As[j])
        ya = Ys[j][(2 * np.arange(n)).clip(0, len(Ys[j]) - 1)].copy()  # aug every 2 SP from t = 1 s, like the song
        ya[:, TARGETS.index("vocals")] = 0.0
        Y += [Ys[j], ya]
        sid += [np.full(len(Xs[j]), j), np.full(n, 1000 + j)]
    return np.concatenate(X), np.concatenate(Y), np.concatenate(sid)


def smooth_gate(p, dt=SP, on=0.6, off=0.4, tau=TAU):
    """The live post-processing: EMA (~tau s) + hysteresis."""
    a = 1 - np.exp(-dt / tau)
    s, g, out = p[0], False, []
    for x in p:
        s += (x - s) * a
        g = s > (off if g else on)
        out.append(g)
    return np.array(out)


def prf(pred, y):
    tp = float((pred & y).sum())
    p = tp / max(pred.sum(), 1)
    r = tp / max(y.sum(), 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def at_labels(t_lab, t_pred, g):
    """Live decision (gate state) at the end of every teacher frame."""
    k = np.searchsorted(t_pred, t_lab + 1e-6) - 1
    out = np.zeros(len(t_lab), bool)
    out[k >= 0] = g[k[k >= 0]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs", action="append", required=True,
                    help="folder of your songs (mp3/wav/m4a/flac/ogg); repeatable for several folders")
    ap.add_argument("--out", default=str(dl.STUDENT_PATH))
    ap.add_argument("--relabel", action="store_true", help="recompute the Demucs teacher labels")
    ap.add_argument("--teacher", default="htdemucs_ft", help="Demucs model of the teacher (htdemucs_ft, htdemucs)")
    ap.add_argument("--model", choices=("lin", "mlp"), default="lin", help="student model per target")
    ap.add_argument("--l2", type=float, default=30.0)
    ap.add_argument("--no-cv", action="store_true", help="skip the leave-one-song-out evaluation")
    ap.add_argument("--report", help="write the cross-validation report (json) here")
    args = ap.parse_args(argv)
    folders = [str(Path(f).expanduser()) for f in args.songs]
    paths = sorted({p for f in folders for ext in ("mp3", "wav", "m4a", "flac", "ogg")
                    for p in glob.glob(os.path.join(os.path.expanduser(f), f"*.{ext}"))})
    if len(paths) < 3:
        raise SystemExit("need at least 3 songs")
    clf = dl.make_yamnet()
    emb = dl.make_yamnet_embedder()
    names = dl.yamnet_names(clf)
    data = []
    for p in paths:
        key = song_key(p)
        lab, aug = teacher_labels(p, key, clf, emb, names, args.teacher, args.relabel)
        fe = student_features(p, key, clf, emb)
        # student step at t sees audio up to t: target = the teacher frame holding t - 0.01 s
        idx = np.searchsorted(lab["t"], fe["t"] - 1e-6).clip(0, len(lab["t"]) - 1)
        Y = np.stack([lab[f"y_{s}"][idx] for s in TARGETS], 1).astype(float)
        data.append({"name": Path(p).name, "fe": fe, "Y": Y, "lab": lab, "aug": aug})
        print(f"  {Path(p).name[:40]:40s} steps {len(Y):4d}  "
              + "  ".join(f"{s} {100 * lab[f'y_{s}'].mean():3.0f}%" for s in TARGETS), flush=True)

    report = {"songs": {}, "mean": {}, "label_version": LABEL_VERSION, "feat_version": FEAT_VERSION,
              "teacher": args.teacher, "model": args.model, "tau": TAU}
    heads = ("full", "lite")
    if not args.no_cv:
        print(f"\nleave-one-song-out ({args.model}, EMA {TAU} s + hysteresis like live): precision / recall / F1")
        for h in heads:
            Xs = [head_inputs(d["fe"], h) for d in data]
            As = [head_inputs(d["aug"], h) for d in data]
            for i, d in enumerate(data):
                tr = [j for j in range(len(data)) if j != i]
                Xtr, Ytr, sid = training_set(Xs, As, [dd["Y"] for dd in data], tr)
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                row = {}
                for k, s in enumerate(TARGETS):
                    m = fit(args.model, (Xtr - mu) / sd, Ytr[:, k], sid, args.l2)
                    p = predict(args.model, m, (Xs[i] - mu) / sd)
                    g = at_labels(d["lab"]["t"], d["fe"]["t"], smooth_gate(p, on=TH_ON[s], off=TH_OFF[s]))
                    y = d["lab"][f"y_{s}"].astype(bool)
                    row[s] = prf(g, y) + (float(y.mean()), float(g.mean()))
                report["songs"].setdefault(d["name"], {})[h] = row
                print(f"  {h} {d['name'][:30]:30s} " + "  ".join(f"{s[:3]} {r[0]:.2f}/{r[1]:.2f}/{r[2]:.2f}"
                                                                for s, r in row.items()), flush=True)
            # mean over the songs where the target is present (>= 5 % of frames); false-on elsewhere
            report["mean"][h] = {s: [float(np.mean([r[h][s][k] for r in report["songs"].values() if r[h][s][3] >= 0.05]
                                               or [0.0])) for k in range(3)] for s in TARGETS}
            fo = {s: [r[h][s][4] for r in report["songs"].values() if r[h][s][3] < 0.05] for s in TARGETS}
            print(f"  {h} mean F1 (songs with the target): " + ", ".join(f"{s} {v[2]:.2f}" for s, v in report["mean"][h].items())
                  + " | on-time where absent: " + ", ".join(f"{s} {100 * np.mean(v):.1f}%" for s, v in fo.items() if v))

    # final models on every song
    save = {"targets": np.array(TARGETS), "n_songs": len(data), "version": dl.STUDENT_VERSION, "tau": TAU,
            "th_on": np.array([TH_ON[s] for s in TARGETS]), "th_off": np.array([TH_OFF[s] for s in TARGETS]),
            "label_version": LABEL_VERSION, "feat_version": FEAT_VERSION}
    for h in heads:
        X, Y, sid = training_set([head_inputs(d["fe"], h) for d in data], [head_inputs(d["aug"], h) for d in data],
                                 [d["Y"] for d in data], range(len(data)))
        mu, sd = X.mean(0), X.std(0) + 1e-6
        ms = [fit(args.model, (X - mu) / sd, Y[:, k], sid, args.l2) for k in range(len(TARGETS))]
        save[f"{h}_kind"], save[f"{h}_mu"], save[f"{h}_sd"] = args.model, mu, sd
        for name in ms[0]:
            save[f"{h}_{name}"] = np.array([m[name] for m in ms])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.copy2(out, out.with_name(out.stem + ".prev" + out.suffix))
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **save)
    tmp.replace(out)
    print(f"model: {out} ({len(data)} songs + their instrumental versions, {len(Y)} steps; previous kept as {out.stem}.prev{out.suffix})")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
