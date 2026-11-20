"""dance_music.py - turns a song (mp3/wav/...) into a choreography for tools/dance.py.

    analyze(path) -> beat times (s), song duration, energy of each beat (0..1)
    build_routine(energy) -> [(move, beats, amp)], one move per 8-beat phrase

The beats come from librosa's beat tracker (the real beats of the song, so the moves stay on the
music even if the tempo drifts). If the song is fast (> --max-dance-bpm), the arm dances on every
other beat (half time). Each 8-beat phrase gets a move from the tier matching its loudness: calm
phrases sway / nod / reach, loud ones bounce / groove; the amplitude follows the loudness too.
The same song always gives the same choreography. Needs librosa (python <= 3.12 for numba):
    uv run --no-project --python 3.12 --with librosa --with mujoco --with opencv-python-headless ...
"""
import hashlib
import random

PHRASE = 8  # beats per move: a multiple of every move's period (2 or 4)
TIERS = {   # calm -> loud
    0: ["sway", "nod", "reach", "bow"],
    1: ["wave", "circle", "twist", "sway"],
    2: ["bounce", "groove", "circle", "wave"],
}


def analyze(path, max_dance_bpm=115.0):
    import librosa
    import numpy as np
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    duration = len(y) / sr
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats, dtype=float)
    half = tempo > max_dance_bpm
    if half:
        beats = beats[::2]
    rms = librosa.feature.rms(y=y)[0]
    t = librosa.times_like(rms, sr=sr)
    edges = np.append(beats, duration)
    energy = np.array([rms[(t >= a) & (t < b)].mean() if np.any((t >= a) & (t < b)) else 0.0
                       for a, b in zip(edges[:-1], edges[1:])])
    lo, hi = np.percentile(energy, 10), np.percentile(energy, 95)
    energy = np.clip((energy - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return {"beat_times": beats, "duration": duration, "energy": energy, "tempo": tempo,
            "dance_bpm": tempo / (2 if half else 1), "half_time": half}


def build_routine(energy, seed_text=""):
    """One move per 8-beat phrase; the last phrase is a bow."""
    import numpy as np
    rng = random.Random(int(hashlib.md5(seed_text.encode()).hexdigest()[:8], 16))
    n = len(energy) // PHRASE
    phrase_e = np.array([energy[k * PHRASE:(k + 1) * PHRASE].mean() for k in range(n)])
    if n == 0:
        return []
    # tiers by rank inside the song (a quiet song still uses all its moves)
    cuts = np.percentile(phrase_e, [33, 66])
    routine, prev = [], None
    for k, e in enumerate(phrase_e):
        if k == n - 1:
            name = "bow"
        else:
            tier = int(e > cuts[0]) + int(e > cuts[1])
            options = [m for m in TIERS[tier] if m != prev] or TIERS[tier]
            name = rng.choice(options)
        amp = 0.55 + 0.45 * float(e)
        routine.append((name, PHRASE, round(amp, 2)))
        prev = name
    return routine
