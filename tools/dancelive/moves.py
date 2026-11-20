"""The move library: beat envelopes (hit, punch, hit_env), the style-specific moves, the event poses,
the dance STYLES (combos per tier, bar accents, amplitude range, tempo feel) and the per-move
derivative tables the speed / acceleration budget uses."""
import math

import numpy as np

import dance
from dance import JOINTS, MOVES, smooth01

# ================================================================ motion
def hit(beat):
    """1 exactly on every beat, 0 half-way between: a smooth "hit" (period 1 beat, so also 2 and 4)."""
    return (1 + math.cos(2 * math.pi * beat)) / 2


_SMOOTH = [False]  # set while a smooth ("_s") variant of a move is evaluated: punch -> hit


def punch(beat, attack=0.4, hold=0.0):
    """A snappy beat envelope (period 1 beat, C2-continuous): min-jerk attack over the last `attack`
    of the beat, peak 1 ON the beat, then a controlled min-jerk release (no hold: fluid, the arm
    never stops). In a smooth variant it is the cosine hit (same peak on the beat, half the speed)."""
    if _SMOOTH[0]:
        return hit(beat)
    p = beat % 1.0
    if p < hold:
        return 1.0
    if p < 1.0 - attack:
        return 1.0 - smooth01((p - hold) / (1.0 - attack - hold))
    return smooth01((p - (1.0 - attack)) / attack)


def hit_env(x, attack=0.3):
    """One big hit over 4 beats (x in beats from the hit): `attack`-beat attack ending ON the hit,
    then straight into a smooth release by beat 3 (no hold: the arm flows through the hit).
    Defined for x in [-attack, 3), 0 elsewhere."""
    if x < -attack or x >= 3.0:
        return 0.0
    if x < 0.0:
        return smooth01((x + attack) / attack)
    return 1.0 - smooth01(x / 3.0)


W, OB, pulse = dance.wave, dance.ON_BEAT, dance.pulse
# Style-specific moves, same contract as dance.MOVES (offsets in rad at amp 1, periods 1, 2 or 4 beats,
# hits on the beats). At amp 1 each one (and its left/right mirror) stays >= 0.15 rad inside the joint
# limits and passes dance.check_collisions: python tools/dance_live.py --check-moves.
STYLE_MOVES = {
    # electronic: a 4-on-the-floor pump (whole arm slams down on every beat), a sweeping "laser",
    # and a "rave" jump with the forearm thrown up
    "pump":     lambda b: {"shoulder_lift": 0.26 * punch(b), "elbow_flex": -0.52 * punch(b),
                           "wrist_flex": 0.30 * punch(b), "shoulder_pan": 0.40 * W(b, 4)},
    "laser":    lambda b: {"shoulder_pan": 0.90 * W(b, 4), "wrist_roll": 0.95 * W(b, 4, 0.25),
                           "wrist_flex": 0.30 * punch(b)},
    "rave":     lambda b: {"shoulder_lift": 0.32 * punch(b), "elbow_flex": -0.65 * punch(b),
                           "wrist_flex": -0.32 * punch(b), "shoulder_pan": 0.45 * W(b, 2, OB),
                           "wrist_roll": 0.55 * W(b, 2, OB)},
    # hip-hop: heavy head-nod on every beat with a pan, a laid-back body bop
    "headnod":  lambda b: {"wrist_flex": 0.55 * punch(b), "elbow_flex": -0.20 * punch(b),
                           "shoulder_pan": 0.28 * W(b, 4)},
    "bop":      lambda b: {"shoulder_lift": 0.20 * punch(b), "elbow_flex": -0.40 * punch(b),
                           "wrist_flex": 0.40 * W(b, 2, OB), "shoulder_pan": 0.40 * W(b, 4)},
    # rock: headbang (wrist + forearm slam on every beat) and a wide thrash
    "headbang": lambda b: {"wrist_flex": 0.60 * punch(b), "elbow_flex": -0.34 * punch(b),
                           "shoulder_lift": 0.16 * punch(b)},
    "thrash":   lambda b: {"shoulder_lift": 0.24 * W(b, 2, OB), "elbow_flex": -0.52 * W(b, 2, OB),
                           "wrist_flex": 0.52 * W(b, 2, OB), "shoulder_pan": 0.60 * W(b, 4)},
    # latin: hip sway (pan) with the wrist rolling against it, and a salsa step
    "hipsway":  lambda b: {"shoulder_pan": 0.80 * W(b, 4), "wrist_roll": -0.60 * W(b, 2, OB),
                           "wrist_flex": 0.20 * W(b, 2, OB)},
    "salsa":    lambda b: {"shoulder_pan": 0.55 * W(b, 4), "wrist_roll": 0.95 * W(b, 4, 0.25),
                           "shoulder_lift": 0.16 * W(b, 2, OB), "wrist_flex": 0.30 * W(b, 2, OB + 0.25)},
    # pop: a big spin and a jump on the beat
    "spin":     lambda b: {"shoulder_pan": 0.80 * W(b, 4), "wrist_roll": 1.05 * W(b, 4, 0.25),
                           "wrist_flex": 0.30 * punch(b)},
    "jump":     lambda b: {"shoulder_lift": 0.30 * punch(b), "elbow_flex": -0.56 * punch(b),
                           "wrist_flex": 0.28 * W(b, 2, OB), "shoulder_pan": 0.35 * W(b, 4)},
    # electronic (more): a left/right strobe on every beat, a dubstep wobble (fast wrist roll over a
    # slow body wave), a shuffle (pump that steps left / right)
    "strobe":   lambda b: {"shoulder_pan": 0.50 * W(b, 2, OB), "wrist_roll": -0.45 * W(b, 2, OB),
                           "wrist_flex": 0.30 * punch(b), "elbow_flex": -0.20 * punch(b)},
    "wobble":   lambda b: {"wrist_roll": 0.32 * W(b, 1, OB), "shoulder_lift": 0.18 * W(b, 4),
                           "elbow_flex": -0.38 * W(b, 4), "shoulder_pan": 0.35 * W(b, 4, 0.25)},
    "shuffle":  lambda b: {"shoulder_pan": 0.38 * W(b, 2, OB), "shoulder_lift": 0.20 * punch(b),
                           "elbow_flex": -0.40 * punch(b), "wrist_flex": 0.26 * punch(b)},
    # hip-hop (more): a wrist chop stepping left / right, a slow lean with the nod, an arm wave
    "chop":     lambda b: {"wrist_flex": 0.45 * punch(b), "shoulder_pan": 0.32 * W(b, 2, OB),
                           "wrist_roll": 0.40 * W(b, 4)},
    "lean":     lambda b: {"shoulder_pan": 0.45 * W(b, 8), "shoulder_lift": 0.16 * W(b, 8, 0.25),
                           "wrist_flex": 0.40 * punch(b), "elbow_flex": -0.18 * punch(b)},
    "armwave":  lambda b: {"shoulder_lift": 0.12 * W(b, 4), "elbow_flex": 0.34 * W(b, 4, 0.08),
                           "wrist_flex": -0.42 * W(b, 4, 0.16), "shoulder_pan": 0.30 * W(b, 4, 0.25)},
    # rock (more): a windmill (big pan + lift circle) and a riff (pan steps with a light headbang)
    "windmill": lambda b: {"shoulder_pan": 0.60 * W(b, 4), "shoulder_lift": 0.20 * W(b, 4, 0.25),
                           "elbow_flex": -0.42 * W(b, 4, 0.25), "wrist_flex": 0.28 * W(b, 4, 0.25)},
    "riff":     lambda b: {"shoulder_pan": 0.42 * W(b, 2, OB), "wrist_flex": 0.42 * punch(b),
                           "elbow_flex": -0.26 * punch(b)},
    # latin (more): cumbia (pan step each beat, rolling wrist) and mambo (forearm push, slow pan)
    "cumbia":   lambda b: {"shoulder_pan": 0.38 * W(b, 2, OB), "wrist_roll": 0.70 * W(b, 4, 0.25),
                           "shoulder_lift": 0.12 * punch(b), "elbow_flex": -0.22 * punch(b)},
    "mambo":    lambda b: {"elbow_flex": -0.34 * W(b, 2, OB), "wrist_flex": 0.30 * W(b, 2, OB + 0.25),
                           "shoulder_pan": 0.60 * W(b, 8), "wrist_roll": -0.40 * W(b, 4)},
    # pop (more): point (thrust forward-up on beat 1, to the side on 3), shimmy (wrist + forearm pops)
    "point":    lambda b: {"shoulder_lift": 0.30 * pulse(b, 2), "elbow_flex": -0.50 * pulse(b, 2),
                           "wrist_flex": -0.25 * pulse(b, 2), "shoulder_pan": 0.50 * W(b, 8)},
    "shimmy":   lambda b: {"wrist_flex": 0.34 * punch(b), "elbow_flex": -0.28 * punch(b),
                           "wrist_roll": 0.60 * W(b, 2, OB), "shoulder_pan": 0.25 * W(b, 4)},
    # calm: a slow, wide wave of the whole arm
    "slowwave": lambda b: {"elbow_flex": 0.45 * W(b, 4), "wrist_flex": -0.50 * W(b, 4, 0.15),
                           "shoulder_pan": 0.45 * W(b, 4, 0.25)},
    # calm (more): a figure-8 (pan at 1x, lift / elbow at 2x: a lemniscate), slow wrist circles,
    # "breathing" (the whole arm rises and settles, head counter-bowing), a reach-and-retract arc
    # (reach out left, retract, reach out right), a flowing wave travelling up the arm
    "figure8":  lambda b: {"shoulder_pan": 0.50 * W(b, 4), "shoulder_lift": 0.14 * W(b, 2),
                           "elbow_flex": -0.28 * W(b, 2), "wrist_flex": 0.20 * W(b, 2, 0.1)},
    "wristcircle": lambda b: {"wrist_flex": 0.35 * W(b, 4), "wrist_roll": 0.70 * W(b, 4, 0.25),
                              "elbow_flex": -0.16 * W(b, 4, 0.25), "shoulder_pan": 0.18 * W(b, 8)},
    "breathe":  lambda b: {"shoulder_lift": -0.16 * pulse(b, 4), "elbow_flex": -0.42 * pulse(b, 4),
                           "wrist_flex": 0.30 * pulse(b, 4), "wrist_roll": 0.30 * W(b, 8)},
    "arc":      lambda b: {"shoulder_pan": 0.55 * W(b, 8), "shoulder_lift": 0.30 * pulse(b, 4),
                           "elbow_flex": -0.50 * pulse(b, 4), "wrist_flex": -0.25 * pulse(b, 4)},
    "flow":     lambda b: {"shoulder_lift": 0.12 * W(b, 4), "elbow_flex": 0.32 * W(b, 4, 0.083),
                           "wrist_flex": -0.42 * W(b, 4, 0.166), "wrist_roll": 0.40 * W(b, 4, 0.25),
                           "shoulder_pan": -0.25 * W(b, 8)},
    # combo fills (last bar of a combo, on top of the move): 8th-note punches / a quick wrist roll
    "fillpunch": lambda b: {j: v * hit(2.0 * b) for j, v in FILL_PUNCH.items()},
    "fillroll": lambda b: {"wrist_roll": 0.50 * W(b, 2), "wrist_flex": 0.22 * W(b, 2, 0.25)},
    # events (not chosen as moves, checked like them): the big drop hit pose (arm thrown up), the
    # build-up riser (pan sweep + rising pump), the singer (lean forward, head up, roll phrase)
    "hitpose":  lambda b: {j: v * hit_env(b % 4.0) for j, v in HIT_POSE.items()},
    "riser":    lambda b: {"shoulder_pan": 0.60 * W(b, 2, OB), "shoulder_lift": 0.20 * hit(b),
                           "elbow_flex": -0.40 * hit(b)},
    "singer":   lambda b: {"shoulder_lift": SINGER_LEAN["shoulder_lift"], "wrist_flex": SINGER_LEAN["wrist_flex"],
                           "elbow_flex": SINGER_LEAN["elbow_flex"], "wrist_roll": 0.60 * math.sin(2 * math.pi * b / 8)},
    "finale":   lambda b: {j: v * hit(b / 4.0 + 0.5) for j, v in FINALE_POSE.items()},
}
HIT_POSE = {"shoulder_lift": -0.35, "elbow_flex": -0.70, "wrist_flex": -0.50}   # arm thrown up and back
SINGER_LEAN = {"shoulder_lift": 0.20, "elbow_flex": -0.15, "wrist_flex": -0.20}  # lean forward, head up
FINALE_POSE = {"shoulder_lift": 0.32, "wrist_flex": 0.40}                         # a deep bow
FILL_PUNCH = {"shoulder_lift": 0.16, "elbow_flex": -0.30, "wrist_flex": 0.24}     # 8th-note fill
EVENT_MOVES = ("hitpose", "riser", "singer", "finale", "fillpunch", "fillroll")
ALL_MOVES = {**MOVES, **STYLE_MOVES}
MIRRORED = ("shoulder_pan", "wrist_roll")  # left/right mirror of a move: these two change sign

# Variants of every dance move (checked like the moves, --check-moves): "m+" = WIDE_GAIN x the
# amplitude (bigger, fuller-range: chosen when the speed budget allows, e.g. slow songs), "m_s" =
# smooth (its punched beats become cosine hits: same peaks ON the beats, a continuous curve at half
# the speed demand: keeps the music's tempo when a punched move is too fast for the joint limits),
# "m_s+" = both; "m++" / "m_s++" = XL_GAIN x (full-range sweeps, mostly slow songs) for the moves
# whose 2.2x amplitude still passes the limit / collision checks (2.0 keeps a margin). All 1.6x
# amplitudes pass; 1.5 keeps a margin.
WIDE_GAIN, XL_GAIN = 1.5, 2.0
XL_EXCLUDED = ("bow", "laser", "mambo", "slowwave")  # their 2.2x fails (bow: gripper near the table)
DANCE_MOVES = [m for m in ALL_MOVES if m not in EVENT_MOVES]


def _wide(f, g=WIDE_GAIN):
    return lambda b: {j: g * v for j, v in f(b).items()}


def _smooth(f):
    def s(b):
        _SMOOTH[0] = True
        try:
            return f(b)
        finally:
            _SMOOTH[0] = False
    return s


def _uses_punch(f):
    return any(f(x) != _smooth(f)(x) for x in np.arange(0.05, 4.0, 0.1))


SMOOTHABLE = {m for m in DANCE_MOVES if _uses_punch(ALL_MOVES[m])}
XL = {m for m in DANCE_MOVES if m not in XL_EXCLUDED}
for _m in DANCE_MOVES:
    ALL_MOVES[_m + "+"] = _wide(ALL_MOVES[_m])
    if _m in XL:
        ALL_MOVES[_m + "++"] = _wide(ALL_MOVES[_m], XL_GAIN)
    if _m in SMOOTHABLE:
        ALL_MOVES[_m + "_s"] = _smooth(ALL_MOVES[_m])
        ALL_MOVES[_m + "_s+"] = _wide(_smooth(ALL_MOVES[_m]))
        if _m in XL:
            ALL_MOVES[_m + "_s++"] = _wide(_smooth(ALL_MOVES[_m]), XL_GAIN)


def base_move(key):
    """The dance move a variant key ("rave_s+") belongs to."""
    return key.rstrip("+").removesuffix("_s")


def variants(m):
    """The variants of move m, preferred first (strong, then fluid): XL, smooth XL, wide, smooth wide,
    plain, smooth."""
    out = []
    for g in (["++"] if m in XL else []) + ["+", ""]:
        out.append(m + g)
        if m in SMOOTHABLE:
            out.append(m + "_s" + g)
    return out


def mirror(o):
    return {j: (-v if j in MIRRORED else v) for j, v in o.items()}


# Dance styles. Each has CHOREOGRAPHIES: combos of moves over 16-32 beats (segment = (move, beats,
# mirrored)), per loudness tier (calm / normal / drop section), written with an A/B structure
# (A B A~ B~ call-and-response, A A~ B mirrored repeats...); a bar accent pattern (bass accent
# weight on beats 1-2-3-4: backbeat for rock / hip-hop, four-on-the-floor for electronic...); the
# amplitude range (from the loudness); the tempo feel (half time = every move twice as slow,
# max_bpm = above it the song is danced half time); the moves danced in turn while there is no
# tempo yet (fallbacks, the first one is `fallback`); the fill on the last bar of a combo; the bass
# accent shape and the highs shimmer weight. In a half-time style (calm) the beats are style beats
# (16 style beats = 32 beats of music).
BOUNCE = {"shoulder_lift": 0.15, "elbow_flex": -0.25, "wrist_flex": 0.12}  # "bounce" at amp 1
C = lambda *segs: list(segs)  # noqa: E731  (readability)


def CR(a, b, n=4, mb=1):  # noqa: N802  call-and-response: A B A~ B~ (B mirrored too if mb)
    return [(a, n, 0), (b, n, 0), (a, n, 1), (b, n, mb)]


STYLES = {
    "default":    dict(max_bpm=124, fallbacks=("sway", "circle", "twist"), amp=(0.75, 1.0), half=False,
                       bass=0.15, high=0.10, bar=(1.0, 0.7, 0.9, 0.7), bass_shape=BOUNCE, fill="fillroll",
                       combos={0: [C(("sway", 8, 0), ("nod", 4, 0), ("reach", 4, 0)), C(("wave", 8, 0), ("bow", 8, 0)),
                                   C(("circle", 8, 0), ("sway", 8, 1)), C(("twist", 4, 0), ("twist", 4, 1), ("wave", 8, 0)),
                                   CR("reach", "sway")],
                               1: [C(("groove", 8, 0), ("twist", 4, 0), ("twist", 4, 1)),
                                   C(("wave", 4, 0), ("wave", 4, 1), ("circle", 8, 0)),
                                   C(("bounce", 8, 0), ("sway", 8, 0)), CR("circle", "groove"),
                                   CR("nod", "twist", mb=1) + [("groove", 8, 0)]],
                               2: [C(("bounce", 4, 0), ("groove", 4, 0), ("bounce", 4, 0), ("groove", 4, 1)),
                                   C(("groove", 8, 0), ("wave", 4, 0), ("wave", 4, 1)),
                                   C(("bounce", 8, 0), ("circle", 8, 0)), CR("twist", "bounce", mb=0),
                                   C(("circle", 8, 0), ("bounce", 4, 0), ("groove", 4, 1))]}),
    "electronic": dict(max_bpm=160, fallbacks=("laser", "wobble", "circle"), amp=(0.85, 1.0), half=False,
                       bass=0.25, high=0.12, bar=(1.0, 1.0, 1.0, 1.0), fill="fillpunch",
                       bass_shape={"shoulder_lift": 0.26, "elbow_flex": -0.52, "wrist_flex": 0.30},
                       combos={0: [C(("laser", 8, 0), ("wobble", 8, 0)), CR("circle", "wobble"),
                                   C(("twist", 4, 0), ("twist", 4, 1), ("laser", 8, 1)),
                                   C(("wobble", 8, 0), ("groove", 4, 0), ("groove", 4, 1)),
                                   C(("figure8", 8, 0), ("laser", 8, 1))],
                               1: [C(("pump", 4, 0), ("laser", 4, 1), ("rave", 8, 0)),
                                   CR("strobe", "pump", mb=0), C(("shuffle", 8, 0), ("wobble", 4, 0), ("shuffle", 4, 1)),
                                   C(("groove", 8, 0), ("strobe", 4, 0), ("strobe", 4, 1)),
                                   CR("laser", "shuffle") + [("circle", 8, 0)],
                                   C(("wobble", 4, 0), ("pump", 4, 0), ("wobble", 4, 1), ("rave", 4, 0))],
                               2: [C(("rave", 4, 0), ("pump", 4, 0), ("rave", 4, 1), ("laser", 4, 1)),
                                   C(("pump", 8, 0), ("strobe", 8, 0)),
                                   C(("rave", 2, 0), ("rave", 2, 1), ("jump", 4, 0), ("shuffle", 8, 0)),
                                   C(("shuffle", 4, 0), ("shuffle", 4, 1), ("rave", 4, 0), ("strobe", 4, 1)),
                                   CR("jump", "strobe") + [("pump", 8, 0)]]}),
    "hiphop":     dict(max_bpm=130, fallbacks=("bop", "lean", "armwave"), amp=(0.8, 1.0), half=False,
                       bass=0.25, high=0.04, bar=(0.6, 1.0, 0.6, 1.0), fill="fillpunch",
                       bass_shape={"wrist_flex": 0.55, "elbow_flex": -0.20},
                       combos={0: [C(("headnod", 8, 0), ("armwave", 8, 0)), C(("lean", 8, 0), ("headnod", 8, 1)),
                                   CR("armwave", "nod", mb=0), C(("bop", 8, 0), ("lean", 8, 1)),
                                   C(("groove", 8, 0), ("chop", 8, 0))],
                               1: [CR("headnod", "bop"), C(("bop", 8, 0), ("groove", 8, 0)),
                                   C(("chop", 4, 0), ("chop", 4, 1), ("lean", 8, 0)),
                                   C(("lean", 8, 0), ("bop", 4, 0), ("chop", 4, 1)),
                                   C(("armwave", 8, 0), ("headnod", 4, 0), ("headnod", 4, 1)), CR("groove", "chop")],
                               2: [C(("bop", 4, 0), ("jump", 4, 0), ("bop", 4, 1), ("headnod", 4, 1)),
                                   C(("headnod", 2, 0), ("headnod", 2, 1), ("jump", 4, 0), ("bop", 8, 0)),
                                   CR("chop", "jump"), C(("bounce", 8, 0), ("chop", 8, 0)),
                                   C(("jump", 8, 0), ("lean", 8, 0))]}),
    "rock":       dict(max_bpm=160, fallbacks=("wave", "windmill", "thrash"), amp=(0.85, 1.0), half=False,
                       bass=0.25, high=0.06, bar=(0.7, 1.0, 0.7, 1.0), fill="fillpunch",
                       bass_shape={"wrist_flex": 0.60, "elbow_flex": -0.34, "shoulder_lift": 0.16},
                       combos={0: [C(("wave", 8, 0), ("headnod", 8, 0)), C(("thrash", 8, 0), ("wave", 8, 1)),
                                   C(("windmill", 8, 0), ("riff", 8, 0)), CR("windmill", "wave"),
                                   C(("riff", 8, 0), ("thrash", 8, 1))],
                               1: [CR("headbang", "thrash"), C(("thrash", 8, 0), ("jump", 4, 0), ("headbang", 4, 0)),
                                   C(("riff", 4, 0), ("riff", 4, 1), ("windmill", 8, 0)),
                                   C(("windmill", 8, 0), ("headbang", 4, 0), ("riff", 4, 1)),
                                   C(("riff", 4, 0), ("thrash", 4, 0), ("riff", 4, 1), ("jump", 4, 0))],
                               2: [C(("headbang", 8, 0), ("jump", 4, 0), ("jump", 4, 1)),
                                   C(("thrash", 4, 0), ("thrash", 4, 1), ("headbang", 8, 0)),
                                   C(("jump", 2, 0), ("headbang", 2, 0), ("jump", 2, 1), ("headbang", 2, 1),
                                     ("thrash", 8, 0)),
                                   C(("riff", 4, 0), ("jump", 4, 0), ("riff", 4, 1), ("windmill", 4, 0)),
                                   CR("windmill", "headbang")]}),
    "pop":        dict(max_bpm=160, fallbacks=("sway", "wave", "twist"), amp=(0.8, 1.0), half=False,
                       bass=0.18, high=0.10, bar=(1.0, 0.6, 0.8, 0.6), bass_shape=BOUNCE, fill="fillpunch",
                       combos={0: [C(("wave", 8, 0), ("spin", 8, 0)), C(("sway", 8, 0), ("reach", 4, 0), ("nod", 4, 0)),
                                   C(("point", 8, 0), ("shimmy", 8, 0)), CR("twist", "wave"),
                                   C(("shimmy", 4, 0), ("sway", 4, 0), ("shimmy", 4, 1), ("reach", 4, 0))],
                               1: [C(("spin", 8, 0), ("jump", 4, 0), ("wave", 4, 1)),
                                   C(("groove", 8, 0), ("twist", 4, 0), ("twist", 4, 1)),
                                   C(("wave", 4, 0), ("wave", 4, 1), ("spin", 8, 1)), CR("point", "shimmy"),
                                   C(("shimmy", 8, 0), ("groove", 8, 1)),
                                   C(("point", 8, 0), ("jump", 4, 0), ("bounce", 4, 0))],
                               2: [CR("jump", "spin"), C(("groove", 8, 0), ("jump", 8, 0)),
                                   C(("shimmy", 4, 0), ("jump", 4, 0), ("shimmy", 4, 1), ("point", 4, 0)),
                                   CR("bounce", "shimmy", mb=1) + [("spin", 8, 0)],
                                   C(("jump", 8, 0), ("point", 8, 1))]}),
    "latin":      dict(max_bpm=130, fallbacks=("hipsway", "mambo", "sway"), amp=(0.8, 1.0), half=False,
                       bass=0.12, high=0.10, bar=(1.0, 0.5, 0.8, 0.5), fill="fillroll",
                       bass_shape={"wrist_roll": 0.60, "shoulder_pan": 0.25},
                       combos={0: [C(("hipsway", 8, 0), ("hipsway", 8, 1)), C(("circle", 8, 0), ("hipsway", 8, 1)),
                                   C(("mambo", 8, 0), ("sway", 8, 0)),
                                   C(("cumbia", 8, 0), ("twist", 4, 0), ("twist", 4, 1)),
                                   C(("sway", 8, 0), ("mambo", 8, 1))],
                               1: [C(("hipsway", 8, 0), ("salsa", 8, 1)), C(("salsa", 8, 0), ("circle", 8, 0)),
                                   CR("cumbia", "salsa"), C(("mambo", 8, 0), ("cumbia", 8, 0)),
                                   CR("hipsway", "mambo")],
                               2: [C(("salsa", 4, 0), ("salsa", 4, 1), ("hipsway", 8, 0)),
                                   C(("spin", 8, 0), ("salsa", 8, 1)),
                                   C(("cumbia", 4, 0), ("cumbia", 4, 1), ("salsa", 4, 0), ("mambo", 4, 1)),
                                   CR("salsa", "cumbia") + [("spin", 8, 0)], C(("mambo", 8, 0), ("salsa", 8, 0))]}),
    "calm":       dict(max_bpm=100, fallbacks=("slowwave", "figure8", "flow", "breathe"), amp=(0.65, 0.9),
                       half=True, bass=0.0, high=0.04, bar=(1.0, 0.3, 0.6, 0.3), bass_shape={}, fill="fillroll",
                       combos={0: [C(("slowwave", 8, 0), ("breathe", 8, 0)), C(("reach", 8, 0), ("bow", 8, 0)),
                                   C(("figure8", 8, 0), ("figure8", 8, 1)), C(("wristcircle", 8, 0), ("sway", 8, 0)),
                                   C(("arc", 8, 0), ("flow", 8, 0)), C(("breathe", 8, 0), ("wristcircle", 8, 1)),
                                   C(("flow", 8, 0), ("slowwave", 8, 1))],
                               1: [C(("slowwave", 8, 0), ("circle", 8, 0)), C(("bow", 8, 0), ("wave", 8, 0)),
                                   C(("figure8", 8, 0), ("arc", 8, 0)), CR("flow", "wristcircle"),
                                   C(("sway", 8, 0), ("figure8", 8, 1)), C(("arc", 8, 0), ("slowwave", 8, 1)),
                                   C(("breathe", 8, 0), ("circle", 8, 1))],
                               2: [C(("wave", 8, 0), ("circle", 8, 0)), C(("slowwave", 8, 0), ("reach", 8, 0)),
                                   C(("figure8", 4, 0), ("figure8", 4, 1), ("arc", 8, 0)),
                                   C(("flow", 8, 0), ("wave", 4, 0), ("wave", 4, 1)), CR("circle", "wristcircle")]}),
}
for _s in STYLES.values():
    _s["fallback"] = _s["fallbacks"][0]
    # every move the style dances (any tier): the pool a combo segment can be swapped from when one
    # move is taking too large a share of the song
    _s["pool"] = tuple(dict.fromkeys(m for tier in _s["combos"].values() for c in tier for m, _, _ in c))


def _derivs(f, period=8.0, h=0.005):
    """Per joint: peak |offset| (rad), |d/dbeat| (rad/beat), |d2/dbeat2| (rad/beat^2) of f(beat)."""
    b = np.arange(0.0, period, h)
    q = np.array([[f(x).get(j, 0.0) for j in JOINTS] for x in b])
    d1 = np.diff(q, axis=0) / h
    d2 = np.diff(q, n=2, axis=0) / (h * h)
    return np.abs(q).max(0), np.abs(d1).max(0), np.abs(d2).max(0)


MOVE_D = {m: _derivs(ALL_MOVES[m]) for m in ALL_MOVES}
_MOVE_DV = {}


def move_d(m, div=1):
    """Derivative table of move m danced `div` times slower (f(b / div)): same peak, d/dbeat / div,
    d2/dbeat2 / div^2. div = 2: hits every 2 beats instead of every beat (still on the beats)."""
    if div == 1:
        return MOVE_D[m]
    key = (m, div)
    if key not in _MOVE_DV:
        mx, d1, d2 = MOVE_D[m]
        _MOVE_DV[key] = (mx, d1 / div, d2 / (div * div))
    return _MOVE_DV[key]
MOVE_SPEED = {m: float(MOVE_D[m][1].max()) for m in ALL_MOVES}  # rad/beat, any joint
HIGH_SHAPE = {"wrist_flex": 0.5 * 0.30, "wrist_roll": 0.5 * 0.48}  # half "nod" + half "twist"
HIGH_D = _derivs(lambda b: {"wrist_flex": HIGH_SHAPE["wrist_flex"] * math.sin(2 * math.pi * b),
                            "wrist_roll": HIGH_SHAPE["wrist_roll"] * math.cos(2 * math.pi * b)}, period=1.0)
BOUNCE_D = _derivs(lambda b: {j: v * hit(b) for j, v in BOUNCE.items()}, period=1.0)
GTR_D = _derivs(lambda b: {"shoulder_pan": 0.60 * W(b, 4)}, period=4.0)
SOFT_D = _derivs(lambda b: ALL_MOVES["slowwave"](b / 2.0), period=16.0)
HIT_D = _derivs(lambda b: {j: v * hit_env(b - 0.5) for j, v in HIT_POSE.items()}, period=4.5)

