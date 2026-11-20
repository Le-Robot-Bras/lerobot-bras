"""Choreography: the Dancer turns the beat clock + analysis (+ style, vocals, instruments) into joint
targets: combos, phrase / switch logic, musical events (drop, build, break, finale), accents, the
singer and the mouth."""
import math
import random

from dance import BASE, JOINTS, smooth01

import numpy as np

from .moves import (ALL_MOVES, BOUNCE, BOUNCE_D, GTR_D, HIGH_D, HIGH_SHAPE, HIT_D, HIT_POSE, FINALE_POSE,
                    MOVE_D, SOFT_D, STYLES, W, _derivs, base_move, hit, hit_env, mirror, move_d, punch, variants)

# Variety: no move should take more than ~25 % of the danced time of a song. A combo whose moves would
# push one of them over SHARE_SOFT is less likely to be picked; a segment whose move is already over
# SHARE_CAP is danced with another move of the style (the same swap for the rest of the combo, so the
# A/B structure and the mirrored repeats stay). Shares are measured over at least SHARE_MIN_T seconds.
SHARE_SOFT, SHARE_CAP, SHARE_MIN_T = 0.17, 0.21, 60.0
# Dynamics inside a combo (amplitude per segment, constant over the segment so no extra slope): the
# first time a move is danced in a combo it is a bit smaller than its repeats; shapes per combo pass.
DYN_SHAPES = ("steady", "crescendo", "answer")
FILL_W = 0.18  # weight of the fill on the last bar (2 beats) of a combo
# ---------------------------------------------------------------- drop hits (rare, meaningful)
# The big "hit" pose is fired by the analysis' fast loudness jump (jump_db: ~0.1 s loudness minus the
# level ~1 s earlier), judged against the SONG's own dynamics. A jump fires only if ALL hold:
#   - it starts (> HIT_ARM_DB, staying above it) and is confirmed HIT_CONFIRM s later (a crash / fill
#     decays) -- or at once if it is already >= HIT_BIG_DB 0.1 s after its start;
#   - it is >= HIT_MIN_DB (calm style: HIT_MIN_DB_CALM) and >= the HIT_JUMP_PCT percentile of jump_db
#     over the last HIT_JUMP_WIN s (with less than HIT_JUMP_MIN_N samples of it: >= HIT_BIG_DB);
#   - the level after (peak of the fast loudness over the confirmation) is in the loudest
#     (1 - HIT_LOUD_RANK) of the song so far (HIT_LOUD_RANK_BIG for a jump >= HIT_BIG_DB);
#   - the HIT_PRE_S s before it were a quieter stretch (mean >= HIT_PRE_DB dB under the level after,
#     peak >= HIT_PRE_PEAK_DB under it): rising out of a break / build, not a dip and return;
#   (the jump statistics skip the first HIT_JUMP_SKIP_S s of a song: its start is not a reference)
#   - >= HIT_REFR_BEATS musical beats and >= HIT_MIN_GAP_S s since the last hit (the gap is waived
#     for a jump >= HIT_BIG_DB), and >= HIT_SONG_WAIT_S s into the song (a new song = a silence gap,
#     a lost / re-acquired tempo lock or a non-octave tempo change; or the analysis' song_t / new_song).
# It lands on the beat nearest to the jump's start: ON it if still ahead, now (short attack) if it
# passed < 0.6 beat ago, else on the next beat. The late section change (calm/normal -> drop) only
# switches combo; it hits only as a fallback: no hit for HIT_MIN_GAP_S, >= HIT_FALLBACK_NONDROP_S s
# of non-drop before, not in the calm style. HITS_ENABLED (Dancer.hits_enabled) turns hits off.
HITS_ENABLED = True
HIT_ARM_DB, HIT_CONFIRM, HIT_MIN_DB, HIT_BIG_DB = 3.0, 0.3, 6.0, 12.0
HIT_JUMP_PCT, HIT_JUMP_WIN, HIT_JUMP_MIN_N, HIT_MIN_DB_CALM = 95.0, 60.0, 100, 8.0
HIT_LOUD_RANK, HIT_LOUD_RANK_BIG, HIT_JUMP_SKIP_S = 0.8, 0.5, 5.0
HIT_PRE_S, HIT_PRE_DB, HIT_PRE_PEAK_DB = 3.0, 6.0, 0.0
HIT_REFR_BEATS, HIT_MIN_GAP_S, HIT_SONG_WAIT_S = 16.0, 30.0, 10.0
HIT_FALLBACK_NONDROP_S = 8.0
# drop_eta is only a hint: held >= 1 s with build_level >= PREARM_BUILD it drives the riser (never
# fires the hit)
PREARM_BUILD = 0.5
# Speed budget: rather than shrinking the whole dance when the tempo and the joint limits (--vmax /
# --amax, e.g. the real arm) cannot follow, the choreography adapts. Each segment's move is checked
# against the budget (with the accents at their minimum): if its scale would be < BUDGET_FIT it is
# danced half speed (period x2: hits every 2 beats, still on the beats), else swapped for a move of
# the style that fits (full or half speed, least danced first). Every tick the accents (bass punch,
# highs, fill, riser, bounce, guitar sweep) give way first, down to ACC_MIN of their weight, so the
# main move keeps its amplitude.
BUDGET_FIT, BUDGET_TARGET, ACC_MIN = 0.9, 0.95, 0.3
FLOW_W = 0.10  # weight of the constant gentle flow layer (slow figure-8) under the dance
BREAK_BEATS, BREAK_DEPTH = 4.0, 0.6  # a break: 4 beats of slow-down (to 40 %) and catch-up (160 %)
CLOCK_TAU = 0.25  # s: the dance clock absorbs the beat tracker's phase jumps over ~this time
FLOW_D = move_d("figure8", 2)
from .safety import budget_scale
from .state import SECTION_TIER, _ema

SWITCH_REASONS = ("start", "phrase", "section", "style", "tempo", "energy", "drop", "build", "break")


class Dancer:
    """Live beat clock + analysis + dance style -> joint targets (before the safety filter).

    Choreography: the style plays COMBOS (sequences of moves over 16-32 beats, some mirrored
    left/right), chosen per loudness tier; the bass accent follows the style's bar pattern (backbeat,
    four-on-the-floor...). Musical events interrupt it at the next beat (min dwell 2 beats), always
    with a 1-beat crossfade so the joint targets stay continuous:
      section / style / tempo change -> a new combo;
      BUILD-UP (energy rising for 2 s) -> a riser (pan sweep + pump) growing with the build;
      DROP (into the drop section, or an energy jump) -> a big hit pose ON the next beat, then a
          drop combo at full power;
      BREAK (energy collapsing) -> 2 beats of slow motion (never frozen), then a slow combo;
      end of the music -> a finishing bow, then back to the base pose.
    Vocals: while someone sings the arm "sings": leans forward, head up, slow wrist-roll phrase per
    sung line, gripper = mouth; the beat moves continue at reduced power. Instrumental: full power.
    Variety: combos are picked least-recently-used first (randomised, seeded per song by its tempo),
    penalised when a move would take too big a share of the song, and a move over ~21 % of the danced
    time is swapped for a less danced one of the style; repeats inside a combo vary in amplitude
    (crescendo / call-and-response / bigger repeat), a reused combo is mirrored, and the last 2 beats
    of a combo get a fill. Drops: the analysis' fast loudness jump (jump_db, ~0.1 s) fires the hit on
    the beat; the late section change is only a fallback; drop_eta + build_level only pre-arm.
    Fluid and strong: each segment dances the biggest variant of its move that the speed budget allows
    at the music's tempo (XL / wide / plain, punched or smooth), half speed only as a last resort; the
    accents give way first; no holds (punch / hit without hold, breaks = slow-down and catch-up of the
    dance clock), a constant slow figure-8 flow underneath, and the dance clock absorbs the beat
    tracker's phase jumps smoothly.
    Every component is scaled analytically so the summed peak speed / acceleration per joint stays
    under 85 % of --vmax / --amax at the current tempo (the safety filter is only a safety net)."""

    def __init__(self, amp=1.0, vmax=3.5, seed=0, style="default", amax=120.0, gripper=False):
        self.amp_user = min(max(amp, 0.0), 1.0)
        self.vmax, self.amax, self.gripper = vmax, amax, gripper
        self.vox_slow, self.vox_was_on, self.line_start, self.mouth = 0.0, False, 0.0, 0.0
        self._bd = {}
        self.bass_s = self.high_s = 0.0
        self.k_s = None
        self.rng = random.Random(seed)
        self.beat_prev = None
        self.sb = 0.0      # style beat: the dance beat, at half speed in a half-time style
        self.combo, self.combo_name, self.seg, self.c0 = None, "", 0, 0.0
        self.move, self.mir, self.m0, self.mlen = None, False, 0.0, 8
        self.prev = None   # (move, m0, mirrored, dyn, div) of the previous segment, for the crossfade
        self.div = 1       # the current segment's move is danced div times slower (speed budget)
        self.acc_f = 1.0   # accent weight factor (speed budget: the accents give way first)
        self.rest_pred = []  # last tick's non-move components (accents at ACC_MIN): move fit prediction
        self.a_m_last = 0.0
        self.pending = None
        self.presence = 0.0
        self.g = 0.0       # 1 = the style's moves, 0 = the fallback (no tempo yet)
        self.confident = False
        self.cscale = 0.6
        self.amp = 0.5
        self.style = self.style_next = style if style in STYLES else "default"
        self.uses = {}     # combos danced: keeps the dance varied
        self.seed = seed
        self.seeded = False     # the rng is re-seeded per song (seed + the song's tempo) at the first lock
        self.picks = 0
        self.last_pick = {}     # combo key -> pick number it was last danced (least recently used first)
        self.move_time = {}     # seconds danced per move (this song): the share guard
        self.subs = {}          # move swaps for the current combo
        self.dyn, self.dyn_shape, self.flip = 1.0, "steady", False
        self.silence_t = 0.0
        self.music_t = 0.0
        self.fill_at = None     # style beat where the current combo's fill starts
        self.drop_sched = None  # style beat of a scheduled drop hit (jump_db), the drop combo starts there
        self.hit_att = 0.3      # attack of the current hit (style beats)
        self.last_hit_th = -1e9  # musical beat (theta) of the last drop hit
        self.theta = 0.0
        self.drop_seen = 0.0    # s the analysis has been predicting a drop (drop_eta)
        self.prearmed = False
        self.jump_arm = None    # [t start, theta start, jump 0.1 s after, level before] while confirming
        self.hits_enabled = HITS_ENABLED
        self.hit_dbg = []       # (t, verdict, jump, p95, level rank, dB over the stretch before, over its peak)
        self.last_hit_t = -1e9  # stream time of the last drop hit
        self.song_t = 0.0       # s since the current song started (own heuristic or the analysis')
        self.lf_hist = []       # (t, loud_fast_db) every 0.1 s since the song started
        self.jd_hist = []       # (t, jump_db) every 0.1 s, last HIT_JUMP_WIN s
        self.hist_t = -1.0
        self.nondrop_t = 0.0    # s since the section was last "drop"
        self.unlocked_t = self.unlocked_t_prev = 0.0
        self.song_bpm = 0.0
        self.bps_s = 1.0        # style beats per second (last tick)
        self.move_beats = {}
        self.switches = []  # (t, reason, beats the previous segment lasted)
        self.events = []   # (t, event)
        self.last_section, self.last_bpm, self.was_locked = None, 0.0, False
        self.e_fast = self.e_slow = None
        self.build_t, self.build = 0.0, 0.0
        self.build_own = self.build_ext = 0.0
        self.hit_at = None      # style beat of the scheduled / current drop hit
        self.hold_until = None  # style beat until which the pose is held (break)
        self.hold_b = 0.0
        self.played = 0.0       # seconds of music danced
        self.finale_t = None    # seconds since the finale started
        self.event = ""
        self.inst = {"drums": 1.0, "bass": 0.0, "gtr": 0.0, "soft": 0.0}

    # ------------------------------------------------------------------ choreography
    def _share(self, move, extra_s=0.0, total_extra_s=None):
        """Projected share of the danced time of `move` after `extra_s` more seconds of it."""
        tot = sum(self.move_time.values()) + (extra_s if total_extra_s is None else total_extra_s)
        return (self.move_time.get(move, 0.0) + extra_s) / max(tot, SHARE_MIN_T)

    def _pick_combo(self, section):
        """A combo of the style's tier for this section: never the previous one, least recently used
        first (randomised, seeded per song), and penalised when its moves would take too large a share
        of the song (variety over the song)."""
        tier = SECTION_TIER.get(section, 1)
        combos = STYLES[self.style]["combos"][tier]
        keys = [(self.style, tier, i) for i in range(len(combos))]
        opts = [k for k in keys if k != self.combo_key] or keys
        self.picks += 1
        spb = 1.0 / max(self.bps_s, 0.25)
        w = []
        for k in opts:
            age = min(self.picks - self.last_pick.get(k, -100), 2 * len(combos))
            c = combos[k[2]]
            dur = sum(n for _, n, _ in c) * spb
            per = {}
            for m, n, _ in c:
                per[m] = per.get(m, 0.0) + n * spb
            worst = max(self._share(m, s_, dur) for m, s_ in per.items())
            w.append(age ** 2 * math.exp(-15.0 * max(worst - SHARE_SOFT, 0.0)) * self.rng.uniform(0.6, 1.4))
        k = self.rng.choices(opts, weights=w)[0]
        self.uses[k] = self.uses.get(k, 0) + 1
        self.last_pick[k] = self.picks
        self.combo_key = k
        # a reused combo is danced differently: another dynamics shape, mirrored every other pass
        n_use = self.uses[k]
        self.dyn_shape = DYN_SHAPES[(k[2] + n_use + self.rng.randrange(2)) % len(DYN_SHAPES)]
        self.flip = (n_use % 2 == 0) != (self.rng.random() < 0.25)
        self.subs = {}
        return combos[k[2]]

    def _dyn(self, i):
        """Amplitude of segment i of the combo (0.8..1): repeats bigger than the first time, a
        crescendo over the combo, or a softer call and a bigger response."""
        c = self.combo
        move = c[i][0]
        before = sum(1 for m, _, _ in c[:i] if m == move)
        again = any(m == move for m, _, _ in c[i + 1:])
        if self.dyn_shape == "crescendo":
            d = 0.8 + 0.2 * i / max(len(c) - 1, 1)
        elif self.dyn_shape == "answer":
            d = 1.0 if i % 2 else 0.85
        else:
            d = 1.0
        if before == 0 and again:
            d = min(d, 0.85)  # the first time; the repeat comes back bigger
        elif before > 0:
            d = max(d, 0.95)
        return d

    combo_key = None

    def _request(self, reason):
        if self.pending is None or reason in ("drop", "break"):  # musical events win
            self.pending = reason

    def _segment(self, at):
        move, beats, mir = self.combo[self.seg]
        if move in self.subs:
            move = self.subs[move]
        else:
            # share guard: a move already over the cap this song is swapped for a less danced one
            dur = beats / max(self.bps_s, 0.25)
            if sum(self.move_time.values()) > 20.0 and self._share(move, dur) > SHARE_CAP:
                pool = [m for m in STYLES[self.style]["pool"]
                        if m != move and m != self._base() and m not in self.subs.values()]
                if pool:
                    pool.sort(key=lambda m: self._share(m, dur))
                    alt = self.rng.choice(pool[:3])
                    if self._share(alt, dur) < self._share(move, dur):
                        self.subs[move] = move = alt
        if self.move is not None:
            self.prev = (self.move, self.m0, self.mir, self.dyn, self.div)
        self.dyn = self._dyn(self.seg)
        move, self.div = self._fit(move, beats)
        self.move, self.mlen, self.mir, self.m0 = move, beats, bool(mir) != self.flip, at
        if self.seg == len(self.combo) - 1 and sum(n for _, n, _ in self.combo) >= 16 and beats >= 4:
            self.fill_at = at + beats - 2.0  # the last 2 beats of the combo: a fill
        elif self.fill_at is not None and self.fill_at > at + beats:
            self.fill_at = None

    def _k_pred(self, m, div):
        """Predicted speed-budget scale of move m (danced div times slower) at the current tempo,
        amplitude and limits, the accents at their minimum."""
        w = self.a_m_last * self.dyn
        return budget_scale(self.rest_pred + [(w, None, move_d(m, div))], self.bps_s, self.vmax, self.amax)

    def _base(self):
        return base_move(self.move) if self.move is not None else None

    def _fit(self, move, beats):
        """(variant key, div) for a segment, FLUID and STRONG first, slow last: at the music's tempo
        the move's variants in order (wide, plain, smooth wide, smooth: the first whose budget scale
        is >= BUDGET_FIT), else another move of the style that fits at tempo (least danced first),
        else, as a last resort, half speed (hits every 2 beats), else the best available."""
        if self.a_m_last <= 0.0:
            return move + "+", 1
        for v in variants(move):
            if self._k_pred(v, 1) >= BUDGET_FIT:
                return v, 1
        dur = beats / max(self.bps_s, 0.25)
        pool = [m for m in STYLES[self.style]["pool"] if m != self._base() and self._share(m, dur) <= SHARE_CAP]
        fit = []
        for m in pool:
            for v in variants(m):
                if self._k_pred(v, 1) >= BUDGET_FIT:
                    fit.append((m, v))
                    break
        if fit:
            fit.sort(key=lambda c: self._share(c[0], dur))
            m, v = self.rng.choice(fit[:3])
            self.subs[move] = m
            return v, 1
        for v in reversed(variants(move)):  # last resort: half speed, smoothest first
            if self._k_pred(v, 2) >= BUDGET_FIT:
                return v, 2
        best = max([(v, d) for v in variants(move) for d in (1, 2)], key=lambda c: self._k_pred(*c))
        return best

    def _start(self, at, st, reason, t):
        """A new combo at style beat `at` (an integer beat)."""
        if self.move is not None:
            self.switches.append((t, reason, at - self.m0))
        if self.style_next != self.style:
            self.style = self.style_next
        section = st.section
        if reason == "drop":
            section = "drop"
            if self.hit_at is None or abs(self.hit_at - at) > 1.5:  # (a jump_db hit is already timed)
                self.hit_at, self.hit_att = at + 1.0, 0.5  # on the next beat, with a full attack
            self.last_hit_th = max(self.last_hit_th, self.theta)
            self.last_hit_t = max(self.last_hit_t, t)
            self.drop_sched = None
        elif reason == "break":
            self.hold_until, self.hold_b = at + BREAK_BEATS, at
            section = "calm"
        if self.inst["drums"] < 0.3 and section != "drop":
            section = "calm"  # no drums: fluid combos
        self.combo = self._pick_combo(section)
        self.c0, self.seg = at, 0
        if self.fill_at is not None and self.fill_at > at:
            self.fill_at = None  # interrupted before its last bar: no fill
        self._segment(at)
        self.pending = None
        if reason in ("drop", "build", "break"):
            self.events.append((t, reason))

    # ------------------------------------------------------------------ one tick
    def step(self, st, beat, dt, style=None, t=0.0, vox=None, inst=None):
        """st: BeatState, beat: dance beat at the time the pose is meant for, style: wanted dance
        style (None = keep), t: time (for the logs), vox: {"on", "mouth"}. Returns (q, info)."""
        vox = vox or {"on": False, "mouth": 0.0}
        # instruments (student): drums -> beat power, bass -> bounce, guitar -> wide sweeps,
        # strings / piano without drums -> slow expressive arm. Unknown (no model): neutral.
        if inst is not None:
            fam = inst["fam"]
            for k, v in (("drums", inst["p"]["drums"]), ("bass", inst["p"]["bass"]),
                         ("gtr", min(fam["GTR"] / 0.2, 1.0)),
                         ("soft", min((fam["PIANO"] + fam["STRINGS"]) / 0.2, 1.0) * (1.0 - inst["p"]["drums"]))):
                self.inst[k] += (v - self.inst[k]) * _ema(dt, 1.0)
        playing = st.section != "silence"
        # a new song (music back after >= 4 s of silence): fresh variety statistics, new seed
        self.silence_t = 0.0 if playing else self.silence_t + dt
        self.music_t = self.music_t + dt if playing else 0.0  # s since the music (re)started
        self._track_song(st, dt, t, playing)
        if not playing and self.silence_t >= 4.0 and self.move_time:
            self.move_time, self.last_pick, self.picks, self.seeded = {}, {}, 0, False
        if st.locked and st.music_bpm > 0 and not self.seeded:
            self.rng = random.Random(f"{self.seed}:{round(st.music_bpm)}")  # randomness seeded per song
            self.seeded = True
        if style in STYLES and style != self.style_next:
            self.style_next = style
        if self.style_next != self.style:
            self._request("style")
        if playing and self.last_section not in (None, "silence") and st.section != self.last_section:
            if st.section == "drop" and not self._fallback_hit_ok(t):
                self._request("section")  # a new (drop tier) combo, no hit
            else:
                self._request("drop" if st.section == "drop" else "break" if self.last_section == "drop"
                              else "section")
        if st.locked and st.music_bpm > 0 and (not self.was_locked
                                               or abs(st.music_bpm / max(self.last_bpm, 1.0) - 1) > 0.04):
            if self.was_locked or self.move is not None:
                self._request("tempo")
            self.last_bpm = st.music_bpm
        self.was_locked = st.locked and st.music_bpm > 0
        self.last_section = st.section
        if playing:
            if self.e_fast is None:
                self.e_fast = self.e_slow = st.energy
            self.e_fast += (st.energy - self.e_fast) * _ema(dt, 0.5)
            self.e_slow += (st.energy - self.e_slow) * _ema(dt, 4.0)
            de = self.e_fast - self.e_slow
            if de > 0.3:  # energy jump: a new combo (the jump_db detector decides about hits)
                self._request("section")
                self.e_slow = self.e_fast
            elif de < -0.3:  # a break
                self._request("break")
                self.e_slow = self.e_fast
            # build-up: energy rising steadily (not yet a drop)
            self.build_t = self.build_t + dt if 0.06 < de < 0.3 and st.section != "drop" else 0.0
            if self.build_t > 2.0 and self.build == 0.0 and self.move is not None:
                self._request("build")
            self.played += dt

        # presence: fades in with the music; at the end, a finishing bow first
        if not playing and self.played > 20.0 and self.finale_t is None and self.presence > 0.5:
            self.finale_t = 0.0
            self.events.append((t, "finale"))
        if self.finale_t is not None:
            self.finale_t += dt
            if playing and self.finale_t > 1.0:
                self.finale_t = None  # the music came back: dance on
        in_finale = self.finale_t is not None and self.finale_t < 3.5
        target_presence = 1.0 if (playing or in_finale) else 0.0
        self.presence += (target_presence - self.presence) * _ema(dt, 0.4)

        self.confident = st.locked and st.music_bpm > 0
        self.g += ((1.0 if self.confident else 0.0) - self.g) * _ema(dt, 0.5)
        # beat power: confidence, and drums (no drums: the tempo is trusted less, the dance is more fluid)
        self.cscale = (0.9 + 0.1 * min(max(st.conf / 0.4, 0.0), 1.0)) * (0.85 + 0.15 * self.inst["drums"])
        S = STYLES[self.style]
        lo, hi = S["amp"]
        self.amp += ((lo + (hi - lo) * st.energy) - self.amp) * _ema(dt, 0.7)

        # style beat: advances with the dance beat (never backward), half as fast in half time
        # the dance clock follows the beat tracker smoothly: it runs at the tempo and absorbs the
        # tracker's phase corrections (jumps) over ~CLOCK_TAU (no jerk), snapping only on big jumps
        if self.beat_prev is None:
            self.beat_prev = self.bclock = beat
        rate = st.rate / max(st.div, 1) if st.rate > 0 else 0.0
        err = beat - (self.bclock + rate * dt)
        if abs(err) > 2.0:
            nb = beat
        else:
            nb = self.bclock + rate * dt + err * min(1.0, dt / CLOCK_TAU)
        self.bclock = nb
        db = max(nb - self.beat_prev, 0.0)
        self.beat_prev = max(nb, self.beat_prev)
        sb0 = self.sb
        self.sb += db * (0.5 if S["half"] else 1.0)
        bps = max(st.bpm, 60.0) / 60.0 * (0.5 if S["half"] else 1.0)
        self.bps_s = bps
        crossed = math.floor(self.sb) > math.floor(sb0)
        self._jump_drop(st, beat, 0.5 if S["half"] else 1.0, dt, t)
        if self.move is None:
            self._start(math.floor(self.sb), st, "start", t)
        elif self.drop_sched is not None and self.sb >= self.drop_sched:
            self._start(math.floor(self.sb), st, "drop", t)  # the drop combo, from the hit's beat
        elif self.pending and crossed and self.sb - self.m0 >= 2.0:
            self._start(math.floor(self.sb), st, self.pending, t)  # musical event, on a beat
        elif self.sb - self.m0 >= self.mlen:  # next segment of the combo, or a new combo
            at = self.m0 + self.mlen
            if self.seg + 1 < len(self.combo):
                self.seg += 1
                self._segment(at)
            else:
                self._start(at, st, "phrase", t)
        # build intensity: grows over 8 beats while the energy keeps rising, collapses otherwise; the
        # analysis' own build-up level (if it has one) drives it too, smoothed
        if self.events and self.events[-1][1] == "build" and self.build_t > 0.0:
            self.build_own = min(1.0, self.build_own + db / 8.0)
        else:
            self.build_own = max(0.0, self.build_own - db / 2.0)
        bl = getattr(st, "build_level", None) if self.prearmed else None  # drop_eta + build: a hint
        self.build_ext += ((min(max(bl, 0.0), 1.0) if bl is not None and playing else 0.0)
                           - self.build_ext) * _ema(dt, 0.5)
        want = max(self.build_own, self.build_ext if self.build_ext > 0.25 else 0.0)
        if self.hit_at is not None and -0.5 <= self.sb - self.hit_at < 3.5:
            want = 0.0  # the drop hit takes over
        self.build += min(max(want - self.build, -2.0 * db), 2.0 * db)  # rate-limited (no jump)
        S = STYLES[self.style]
        b = self.sb
        warp = 1.0  # dance-clock speed (break)
        if self.hold_until is not None and b < self.hold_until:
            # break: BREAK_BEATS of breathing time (never frozen: fluid): the dance clock slows down
            # to 1 - BREAK_DEPTH of its speed then surges back and catches up exactly (speed
            # 1 - A sin^3(pi u / 2): velocity- and acceleration-continuous, nothing lost: the moves
            # come out of it on the beats)
            u = (b - self.hold_b) * 4.0 / (self.hold_until - self.hold_b)
            c = math.cos(math.pi * u / 2.0)
            b_eval = b - BREAK_DEPTH * (2.0 / math.pi) * (2.0 / 3.0 - c + c ** 3 / 3.0) * (self.hold_until - self.hold_b) / 4.0
            warp = 1.0 - BREAK_DEPTH * math.sin(math.pi * u / 2.0) ** 3
        else:
            self.hold_until = None
            b_eval = b
        bm = b_eval - self.m0
        w = smooth01(bm)  # 1-beat crossfade from the previous segment

        amp = self.amp
        # vocals: the "singer" (lean forward, head up, one slow roll phrase per sung line) takes up to
        # 30 % of the amplitude budget while someone sings; the gripper is the mouth
        self.vox_slow += ((1.0 if vox["on"] else 0.0) - self.vox_slow) * _ema(dt, 0.6)
        if vox["on"] and not self.vox_was_on:
            self.line_start = b
        self.vox_was_on = vox["on"]
        self.mouth += (vox["mouth"] - self.mouth) * _ema(dt, 0.04)
        # accent levels and the speed scale are smoothed (~0.25 s): an amplitude that jumps would
        # make the joint targets jump (the bass level itself moves every 23 ms)
        self.bass_s += (st.bass - self.bass_s) * _ema(dt, 0.25)
        self.high_s += (st.high - self.high_s) * _ema(dt, 0.25)
        # this beat's weight in the bar; it switches 0.4 beat before the beat, where the punch is 0
        # (switching on the beat, at the punch's peak, made the accent jump)
        bar = S["bar"][int(math.floor(b_eval + 0.4)) % 4]
        hitx = (b - self.hit_at) if self.hit_at is not None else 99.0
        if hitx > 4.0:
            self.hit_at, hitx = None, 99.0
        c_hit = 0.6 * hit_env(hitx, self.hit_att)
        c_r = 0.3 * self.build
        c_v = 0.25 * self.vox_slow
        c_b = min(S["bass"] * self.bass_s * (1.5 if st.section == "drop" else 1.0)
                  * (0.4 + 0.6 * self.inst["drums"]), 0.25)  # drum hits drive the punch accent
        c_bb = 0.10 * self.inst["bass"]                     # bass line -> bounce
        c_g = 0.12 * self.inst["gtr"] * self.inst["drums"]  # guitar -> wide pan sweep
        c_s = 0.15 * self.inst["soft"]                      # strings / piano -> slow expressive arm
        c_h = S["high"] * self.high_s
        # the fill on the last bar of a combo: in over 1 beat, 1 beat, out over 1 beat
        fx = (b_eval - self.fill_at) if self.fill_at is not None else -1.0
        if fx > 3.0:
            self.fill_at, fx = None, -1.0
        c_f = FILL_W * smooth01(fx) * (1.0 - smooth01(fx - 2.0)) if fx >= 0.0 else 0.0
        free = max(1.0 - c_b - c_h - c_v - c_hit - c_r - c_bb - c_g - c_s - c_f - FLOW_W, 0.1)
        a_m = amp * free * self.cscale  # moves + accents + events <= amp <= 1
        a_b, a_h = amp * c_b * self.cscale, amp * c_h * self.cscale
        kick = punch(b_eval) * bar  # bass accent: punched on every beat, weighted by the bar pattern
        sh = 2.0 * math.pi * b_eval  # highs: a wrist circle per beat (fluid; was an eighth-note shimmer)
        fin = hit_env((self.finale_t - 1.2) / 0.8) if self.finale_t is not None else 0.0  # ~2.8 s bow

        # no tempo yet: the style's own slow moves in turn (16 beats each, 2-beat crossfade) at 60 %,
        # on the free clock
        fbs = S["fallbacks"]
        fi = int(math.floor(b_eval / 16.0))
        fb, fbp = fbs[fi % len(fbs)], fbs[(fi - 1) % len(fbs)]
        wf = smooth01((b_eval - 16.0 * fi) / 2.0)
        mv = ALL_MOVES[self.move](bm / self.div)
        main = [(self.g * w * a_m * self.dyn, mirror(mv) if self.mir else mv, move_d(self.move, self.div))]
        if self.prev is not None and w < 1.0:
            pm, pm0, pmir, pdyn, pdiv = self.prev
            po = ALL_MOVES[pm]((b_eval - pm0) / pdiv)
            main.append((self.g * (1.0 - w) * a_m * pdyn, mirror(po) if pmir else po, move_d(pm, pdiv)))
        acc = []  # accents: they give way first when the speed budget is short
        if c_f > 0.0:
            acc.append((self.g * amp * c_f * self.cscale, ALL_MOVES[S["fill"]](b_eval), MOVE_D[S["fill"]]))
        acc.append((self.g * a_b * kick, S["bass_shape"], self._bass_d(S)))
        acc.append((self.g * a_h, {"wrist_flex": HIGH_SHAPE["wrist_flex"] * math.sin(sh),
                                   "wrist_roll": HIGH_SHAPE["wrist_roll"] * math.cos(sh)}, HIGH_D))
        acc.append((self.g * amp * c_r, ALL_MOVES["riser"](b_eval), MOVE_D["riser"]))
        other = [(amp * c_v, ALL_MOVES["singer"](b - self.line_start), MOVE_D["singer"])]
        hitc = [(self.g * amp * c_hit / 0.6 * 0.6, HIT_POSE if c_hit > 0 else {}, HIT_D)]
        # a constant gentle flow under everything (slow figure-8, 8-beat period): never static
        other.append((self.g * amp * FLOW_W, ALL_MOVES["figure8"](b_eval / 2.0), FLOW_D))
        other.append(((1.0 - self.g) * 0.6 * amp * wf, ALL_MOVES[fb](b_eval), MOVE_D[fb]))
        if wf < 1.0 and fbp != fb:
            other.append(((1.0 - self.g) * 0.6 * amp * (1.0 - wf), ALL_MOVES[fbp](b_eval), MOVE_D[fbp]))
        if c_bb + c_g + c_s > 0.0:
            acc.append((self.g * amp * c_bb * hit(b_eval), BOUNCE, BOUNCE_D))
            acc.append((self.g * amp * c_g, {"shoulder_pan": 0.60 * W(b_eval, 4)}, GTR_D))
            other.append((self.g * amp * c_s, ALL_MOVES["slowwave"](b_eval / 2.0), SOFT_D))
        # accents give way first (smoothed: fast down, slow up), down to ACC_MIN of their weight
        bps_w = bps * max(warp, 1.0)  # the break's catch-up runs the dance clock faster
        f_t = self._acc_factor(main + other + hitc, acc, bps_w)
        self.acc_f += (f_t - self.acc_f) * _ema(dt, 0.08 if f_t < self.acc_f else 0.6)
        acc = [(wg * self.acc_f, o, d) for wg, o, d in acc]
        self.rest_pred = [(wg * ACC_MIN / max(self.acc_f, ACC_MIN), None, d) for wg, _, d in acc] + \
            [(wg, None, d) for wg, _, d in other]
        self.a_m_last = self.g * a_m
        comps = main + acc + other + hitc
        if fin > 0.0:
            comps = [(wg * (1.0 - fin), o, d) for wg, o, d in comps]
            comps.append((fin, FINALE_POSE, MOVE_D["finale"]))

        # analytic speed / acceleration budget per joint (85 % of --vmax / --amax at this tempo)
        k_now = budget_scale(comps, bps_w, self.vmax, self.amax)
        self.k_s = k_now if self.k_s is None else self.k_s + (k_now - self.k_s) * _ema(dt, 0.25)
        k = min(self.k_s, k_now * 1.15)
        off = {j: 0.0 for j in JOINTS}
        for wgt, o, _ in comps:
            for j, v in o.items():
                off[j] += k * wgt * v
        s = self.presence * self.amp_user
        q = {j: BASE[j] + s * off[j] for j in JOINTS}
        if self.gripper:  # the mouth: closed 5 %, open up to 80 % with the voice
            q["gripper"] = 5.0 + 75.0 * self.mouth * self.vox_slow * self.presence
        if playing and self.g > 0.5:
            self.move_beats[self._base()] = self.move_beats.get(self._base(), 0.0) + db
        if playing:  # danced time per move (what is shown): the variety guard
            shown = self._base() if self.g > 0.5 else fb
            self.move_time[shown] = self.move_time.get(shown, 0.0) + dt
        self.event = ("FINALE" if fin > 0.05 else "DROP HIT" if c_hit > 0.05 else "BREAK"
                      if self.hold_until is not None else "BUILD-UP" if self.build > 0.05 else
                      "SINGING" if self.vox_slow > 0.5 else "")
        name = f"{'-'.join(self.subs.get(m, m) for m, _, _ in self.combo)}" if self.combo else ""
        return q, {"move": (self._base() + ("~" if self.mir else "")) if self.g > 0.5 else fb, "bm": bm,
                   "variant": self.move,
                   "beat": b, "amp": amp * k, "style": self.style, "len": self.mlen, "k": k,
                   "combo": name, "event": self.event, "div": self.div,
                   "amp_move": k * self.g * w * a_m * self.dyn * self.presence * self.amp_user,
                   "styled": self.g > 0.5 and self.style != "default"}

    def _acc_factor(self, base, acc, bps):
        """Largest accent factor in [ACC_MIN, 1] keeping the budget scale >= BUDGET_TARGET."""
        if not acc:
            return 1.0
        def dem(comps):
            spd = np.zeros(len(JOINTS))
            ac = np.zeros(len(JOINTS))
            for wgt, _, (mx, d1, d2) in comps:  # same formula as safety.budget_scale
                spd += abs(wgt) * (d1 * bps + mx * 1.9 * bps)
                ac += abs(wgt) * (d2 * bps * bps + mx * 6.0 * bps * bps)
            return spd, ac
        bs, ba = dem(base)
        as_, aa = dem(acc)
        ls, la = 0.85 * self.vmax / BUDGET_TARGET, 0.85 * self.amax / BUDGET_TARGET
        f = 1.0
        for b_, a_, lim in ((bs, as_, ls), (ba, aa, la)):
            m = a_ > 1e-9
            if m.any():
                f = min(f, float(((lim - b_[m]) / a_[m]).min()))
        return min(max(f, ACC_MIN), 1.0)

    def _fallback_hit_ok(self, t):
        """May the late section change into "drop" hit? Only as a fallback (see HIT_*)."""
        return (self.hits_enabled and self.drop_sched is None and self.style != "calm"
                and t - self.last_hit_t >= HIT_MIN_GAP_S and self.nondrop_t >= HIT_FALLBACK_NONDROP_S
                and self.song_t >= HIT_SONG_WAIT_S and self.theta - self.last_hit_th >= HIT_REFR_BEATS)

    def _track_song(self, st, dt, t, playing):
        """Seconds since the current song started (the analysis' song_t / new_song if it has them,
        else: a silence gap, a tempo lock lost >= 1 s then re-acquired, or a non-octave tempo change
        > 8 % start a new song), and the song's loudness / jump histories (every 0.1 s)."""
        new = False
        song_t = getattr(st, "song_t", None)
        if getattr(st, "new_song", False):
            new = True
        if not playing:
            new = new or self.song_t > 0.0
        locked = st.locked and st.music_bpm > 0
        self.unlocked_t = 0.0 if locked else self.unlocked_t + dt
        if locked:
            if self.song_bpm <= 0.0 or self.unlocked_t_prev >= 1.0:
                new = new or (self.song_bpm > 0.0 and self.unlocked_t_prev >= 1.0)
                self.song_bpm = st.music_bpm
            else:
                r = st.music_bpm / self.song_bpm
                if min(abs(r - x) for x in (0.5, 2.0 / 3.0, 1.0, 1.5, 2.0)) > 0.08 and abs(r - 1.0) > 0.08:
                    new = True
                    self.song_bpm = st.music_bpm
        self.unlocked_t_prev = self.unlocked_t
        if new:
            self.song_t, self.lf_hist, self.jd_hist, self.jump_arm = 0.0, [], [], None
        elif playing:
            self.song_t += dt
        if song_t is not None:
            self.song_t = float(song_t)
        self.nondrop_t = 0.0 if st.section == "drop" else self.nondrop_t + dt
        if t - self.hist_t >= 0.1 and playing:
            self.hist_t = t
            lf = getattr(st, "loud_fast_db", None)
            if lf is not None and lf > -100.0:
                self.lf_hist.append((t, lf))
                if len(self.lf_hist) > 9000:
                    self.lf_hist.pop(0)
            jd = getattr(st, "jump_db", None)
            if jd is not None and self.song_t >= HIT_JUMP_SKIP_S:
                self.jd_hist.append((t, jd))
                while self.jd_hist and self.jd_hist[0][0] < t - HIT_JUMP_WIN:
                    self.jd_hist.pop(0)

    def _jump_drop(self, st, beat, f, dt, t):
        """The drop hit from the fast loudness jump, judged against the song's own dynamics (see
        HIT_*), snapped to the beat; drop_eta (with a high build_level, held 1 s) only drives the riser."""
        self.theta = beat * max(st.div, 1) + st.off  # musical beat of the pose being computed
        eta = getattr(st, "drop_eta", None)
        self.drop_seen = self.drop_seen + dt if eta is not None else 0.0
        self.prearmed = self.drop_seen >= 1.0 and (getattr(st, "build_level", 0.0) or 0.0) >= PREARM_BUILD
        jd, lf = getattr(st, "jump_db", None), getattr(st, "loud_fast_db", None)
        if jd is None or lf is None or not self.hits_enabled:
            self.jump_arm = None
            return
        if self.drop_sched is not None or st.section == "silence":
            self.jump_arm = None
            return
        if self.jump_arm is None:
            if jd > HIT_ARM_DB:
                pre = [x for tt, x in self.lf_hist if t - HIT_PRE_S - 0.3 <= tt <= t - 0.3]
                self.jump_arm = [t, self.theta, None, pre, lf]
            return
        self.jump_arm[4] = max(self.jump_arm[4], lf)
        t0, th0, j1, pre, lf_peak = self.jump_arm
        if jd <= HIT_ARM_DB:
            self.jump_arm = None  # a blip
            return
        if j1 is None:
            if t - t0 < 0.1 - 1e-6:
                return
            self.jump_arm[2] = j1 = jd
            if j1 < HIT_BIG_DB:
                return  # wait for the confirmation
        elif t - t0 < HIT_CONFIRM - 1e-6:
            return
        self.jump_arm = None
        jump = max(jd, j1)
        if jump < HIT_MIN_DB:
            return
        jds = [x for _, x in self.jd_hist]
        song = np.array([x for _, x in self.lf_hist])
        p_jump = float(np.percentile(jds, HIT_JUMP_PCT)) if len(jds) >= HIT_JUMP_MIN_N else HIT_BIG_DB
        if self.style == "calm":
            p_jump = max(p_jump, HIT_MIN_DB_CALM)
        lf = lf_peak
        rank = float((song < lf).mean()) if len(song) >= 50 else 0.0
        pre_mean = float(np.mean(pre)) if len(pre) >= 10 * (HIT_PRE_S - 1.0) else lf
        pre_peak = float(max(pre)) if pre else lf
        why = ("song start" if self.song_t < HIT_SONG_WAIT_S or self.move is None else
               "not exceptional" if jump < p_jump else
               "not loud" if rank < (HIT_LOUD_RANK_BIG if jump >= HIT_BIG_DB else HIT_LOUD_RANK) else
               "no quiet before" if pre_mean > lf - HIT_PRE_DB or pre_peak >= lf - HIT_PRE_PEAK_DB else
               "refractory" if self.theta - self.last_hit_th < HIT_REFR_BEATS else
               "min gap" if t - self.last_hit_t < HIT_MIN_GAP_S and jump < HIT_BIG_DB else "")
        self.hit_dbg.append((round(t, 2), why or "HIT", round(jump, 1), round(p_jump, 1), round(rank, 2),
                             round(lf - pre_mean, 1), round(lf - pre_peak, 1)))
        del self.hit_dbg[:-200]
        if why:
            return
        # snap to the beat nearest to the jump's start
        k = f / max(st.div, 1)  # style beats per musical beat
        B = round(th0)
        if B > self.theta + 0.1:      # still ahead: ON it
            self.hit_at = self.sb + (B - self.theta) * k
            self.hit_att = min(0.3, self.hit_at - self.sb)
            self.last_hit_th = B
        elif self.theta - B < 0.6:    # just passed: now, short attack
            self.hit_att = 0.2
            self.hit_at = self.sb + self.hit_att
            self.last_hit_th = self.theta
        else:                         # the next beat
            B = math.floor(self.theta) + 1.0
            self.hit_at = self.sb + (B - self.theta) * k
            self.hit_att = min(0.3, max(self.hit_at - self.sb, 0.1))
            self.hit_at = max(self.hit_at, self.sb + self.hit_att)
            self.last_hit_th = B
        self.last_hit_t = t
        self.drop_sched = self.hit_at  # the drop combo starts on the hit's beat

    def _bass_d(self, S):
        key = id(S["bass_shape"])
        if key not in self._bd:
            sh = S["bass_shape"]
            self._bd[key] = _derivs(lambda b: {j: v * punch(b) for j, v in sh.items()}, period=1.0)
        return self._bd[key]
