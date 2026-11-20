"""The application: the 50 Hz Controller (analysis snapshot -> genre / vocals / instruments ->
Dancer -> safety -> pose, status line, log), the analysis thread, the live twin view, and the two
run modes (run_fast: headless on stream time; run_live: real time, twin and/or real arm)."""
import sys
import threading
import time
from dataclasses import replace

import numpy as np

import dance
from dance import BASE, JOINTS

from .analysis import LiveAnalyzer
from .audio_io import decode_file, decode_stereo_like, make_source
from .choreo import Dancer
from .genre import open_genre
from .latency import RunStats, StateDelay, bench_analyzer, budget_rows, lead_terms, output_delay, print_budget, reactive_total
from .moves import ALL_MOVES, STYLES
from .rosbridge import open_stream
from .safety import ContactGuard, SafetyFilter
from .state import SECTIONS
from .vocals import VocalFusion, open_vocals

def analyzer_options(args):
    """LiveAnalyzer keyword options from the CLI flags (--legacy-analysis: the pre-latency-work behaviour)."""
    if getattr(args, "legacy_analysis", False):
        return dict(hop_div=1, accent_ms=1000.0, band_attack=0.01, band_release=0.15, loud_tau=3.0,
                    fast_loud_tau=0.10, section_dwell=1.5, anticipate=args.anticipate, **section_options(args))
    return dict(hop_div=args.hop_div, accent_ms=args.accent_ms, band_attack=args.band_attack,
                band_release=args.band_release, loud_tau=args.loud_tau, fast_loud_tau=args.fast_loud_tau,
                section_dwell=args.section_dwell, anticipate=args.anticipate, **section_options(args))


def section_options(args):
    return dict(drop_dwell=args.drop_dwell, drop_rank=args.drop_rank, drop_over_db=args.drop_over_db,
                song_dist=args.song_dist, song_gap=args.song_gap, drop_max=args.drop_max)


def make_analyzer(args, sr, hop=None):
    an = LiveAnalyzer(sr, hop=hop, max_dance_bpm=args.max_dance_bpm, **analyzer_options(args))
    an.percussive_onset = args.percussive_onset
    return an


# ================================================================ control (shared by all modes)
def bar(x, n=5):
    return "#" * int(round(min(max(x, 0.0), 1.0) * n))


class Controller:
    """One 50 Hz tick: analysis snapshot -> dancer -> safety -> pose. Keeps a log."""

    def __init__(self, analyzer, args, q0, genre=None, vocals=None):
        self.an = analyzer
        self.args = args
        self.genre = genre
        self.vocals = vocals
        self.dancer = Dancer(args.amp, args.vmax, seed=args.seed, style=args.style or "default",
                             amax=args.amax, gripper=args.gripper)
        self.voice = VocalFusion()  # vox_on / song_vox / vox_act / vox_text (properties below)
        self.inst_text = ""
        self.grip = 5.0
        self.style_text = ""
        self.safety = SafetyFilter(q0, args.vmax, args.amax, args.margin)
        self.guard = None
        if args.guard:
            try:
                self.guard = ContactGuard()
            except ImportError:
                print("contact guard disabled (mujoco not installed)")
        self.frozen_until = -1.0
        self.contacts = 0
        # beat-clock lead (s): --latency + input device (added by run_live) + servo lead (--real) - output delay
        self.lead = sum(lead_terms(args).values())
        # with a late output (AirPods) the reactive features are held back to what is heard
        self.state_delay = StateDelay(max(output_delay(args) - 0.03, 0.0))
        self.log = {k: [] for k in ("t", "beat_raw", "beat", "bpm", "conf", "bass", "mid", "high", "energy",
                                    "loud_db", "section", "move", "style", "styled", "theta", "clip", "vox", "vox_env", "k", "combo", "event", "q", "target",
                                    "drop_eta", "build_level", "jump_db", "energy_fast")}
        self.vpeak = 0.0
        self.late_max = 0.0
        self.clips = self.ticks = 0
        self.lates = []
        self.label = ""
        self.genre_text = ""

    def tick(self, ts, dt, now):
        """ts: stream time (s) now. Returns (q, label)."""
        st = self.an.snapshot()
        st = self.state_delay(ts, st)
        beat = st.beat(ts + self.lead)
        if st.drop_t is not None:  # the drop, as seconds ahead of the moment the pose is computed for
            st = replace(st, drop_eta=max(st.drop_t - (ts + self.lead), 0.0))
        want = self.args.style
        g = self.genre.snapshot() if self.genre is not None else None
        if want is None and g is not None:
            want = g["style"]
            if st.section == "calm" and g["prefers_calm"]:
                want = "calm"  # a calm part of the song softens the dance (genre hysteresis kept)
        vox = self._vocals(st, g, dt)
        inst = self._instruments()
        self.inst_text = ("INST " + self.genre.student.tags()) if inst is not None else ""
        target, info = self.dancer.step(st, beat, dt, want, ts, vox, inst)
        if self.args.max_dance_bpm_auto:  # the style decides above which tempo it dances half time
            self.an.max_dance_bpm = STYLES[info["style"]]["max_bpm"]
        q, v = self.safety.propose(target, dt)
        if now < self.frozen_until:
            q, v = dict(self.safety.q), {j: 0.0 for j in JOINTS}
        elif self.guard is not None:
            hit = self.guard.check(q)
            if hit:
                self.contacts += 1
                self.frozen_until = now + self.args.freeze_time
                print(f"  [{ts:6.1f}s] CONTACT ({hit}): holding for {self.args.freeze_time:.0f} s")
                q, v = dict(self.safety.q), {j: 0.0 for j in JOINTS}
        self.safety.commit(q, v)
        self.vpeak = max(self.vpeak, max(abs(x) for x in v.values()))
        if "gripper" in target:  # the mouth, rate-limited (300 %/s)
            self.grip += min(max(target["gripper"] - self.grip, -300.0 * dt), 300.0 * dt)
            q = dict(q, gripper=self.grip)
        fb = info["beat"]
        pulse = "[#####]" if st.locked and (fb % 1.0) < 0.18 else "[     ]"
        move = info["move"] if st.section != "silence" else "-"
        name = info["style"].upper() if info["style"] != "default" else "AUTO"
        # the MUSIC tempo is what a listener taps; the arm may dance it half time (said separately)
        self.style_text = f"{name}   {st.music_bpm:.0f} BPM" if st.music_bpm > 0 else f"{name}   -- BPM"
        if info["event"]:
            self.style_text += f"   {info['event']}"
        half = st.music_bpm > 0 and st.div == 2
        tempo = (f"music {st.music_bpm:.0f} bpm" + (f" (dance {st.bpm:.0f}, half time)" if half else "")
                 if st.music_bpm > 0 else "no tempo yet")
        self.genre_text = (((self.genre.text() + "   |   ") if self.genre is not None else "") + tempo
                           + "   |   " + (self.inst_text or self.vox_text))
        self.label = (f"{info['style'].upper()}  music {st.music_bpm:5.1f} bpm" + (" (half)" if half else "")
                      + f" c{st.conf:.2f} {pulse} {st.section:<7} {move} {int(info['bm']) + 1}/{info['len']}"
                      f"  B:{bar(st.bass):<5} M:{bar(st.mid):<5} H:{bar(st.high):<5} {self.vox_text}"
                      + (f"  {info['event']}" if info["event"] else ""))
        L = self.log
        L["t"].append(ts)
        L["beat_raw"].append(st.beat(ts))
        L["beat"].append(fb)
        for k in ("bpm", "conf", "bass", "mid", "high", "energy", "loud_db"):
            L[k].append(getattr(st, k))
        L["section"].append(SECTIONS.index(st.section))
        L["move"].append(list(ALL_MOVES).index(info["move"].rstrip("~")))
        L["style"].append(list(STYLES).index(info["style"]))
        L["styled"].append(info["styled"])
        L["vox"].append(self.vox_on)
        L["vox_env"].append(st.vox_env)
        L["k"].append(info["k"])
        L["combo"].append(info["combo"])
        L["event"].append(info["event"])
        L["theta"].append(st.theta_ref + (ts - st.t_ref) * st.rate)  # musical beat clock
        L["drop_eta"].append(np.nan if st.drop_eta is None else st.drop_eta)
        L["build_level"].append(st.build_level)
        L["jump_db"].append(st.jump_db)
        L["energy_fast"].append(st.energy_fast)
        # limiter active: the commanded pose deviates from the (limit-clamped) target
        self.clips += max(abs(q[j] - self.safety.clamp(j, target[j])) for j in JOINTS) > 0.01
        self.ticks += 1
        L["clip"].append(max(abs(q[j] - self.safety.clamp(j, target[j])) for j in JOINTS))
        L["q"].append([q[j] for j in JOINTS])
        L["target"].append([target[j] for j in JOINTS])
        return q, st, info

    def _instruments(self):
        if self.genre is None or self.genre.student is None:
            return None
        return self.genre.student.snapshot()

    def _vocals(self, st, g, dt):
        """Vocals on/off, mouth opening 0..1 and kind: see vocals.VocalFusion."""
        student = self.genre.student if self.genre is not None and self.genre.student else None
        return self.voice.update(st, g, dt, student, self.vocals)

    # the vocal state lives in self.voice (vocals.VocalFusion); kept readable / writable here
    vox_on = property(lambda self: self.voice.vox_on, lambda self, v: setattr(self.voice, "vox_on", v))
    song_vox = property(lambda self: self.voice.song_vox, lambda self, v: setattr(self.voice, "song_vox", v))
    vox_act = property(lambda self: self.voice.vox_act, lambda self, v: setattr(self.voice, "vox_act", v))
    vox_text = property(lambda self: self.voice.vox_text, lambda self, v: setattr(self.voice, "vox_text", v))

    def status_line(self, ts, st, info):
        return (f"[{ts:6.1f}s] music {st.music_bpm:5.1f} bpm (dance {st.bpm:5.1f}) conf {st.conf:.2f} "
                f"{st.section:<7} {info['move']:<6} amp {info['amp']:.2f}  B {st.bass:.2f} M {st.mid:.2f} "
                f"H {st.high:.2f}  loud {st.loud_db:5.1f} dB  speed scale {info['k']:.2f}  limiter active "
                f"{100 * self.clips / max(self.ticks, 1):.1f} % of ticks (peak {self.vpeak:.2f} rad/s)"
                f"  | style {info['style']}  {self.vox_text}  {self.inst_text}" + (f"  {self.genre.text()}" if self.genre is not None else ""))

    def save_log(self, path):
        sw = self.dancer.switches
        ev = self.dancer.events
        np.savez(path, joints=np.array(JOINTS), moves=np.array(list(ALL_MOVES)), sections=np.array(SECTIONS),
                 styles=np.array(list(STYLES)), switch_t=np.array([x[0] for x in sw]),
                 switch_reason=np.array([x[1] for x in sw]), switch_beats=np.array([x[2] for x in sw]),
                 event_t=np.array([x[0] for x in ev]), event_kind=np.array([x[1] for x in ev]),
                 move_beats_names=np.array(list(self.dancer.move_beats)),
                 move_beats=np.array(list(self.dancer.move_beats.values())),
                 song_reset_t=np.array([r[0] for r in self.an.song_resets]),
                 song_reset_why=np.array([r[1] for r in self.an.song_resets]),
                 **{k: np.array(v) for k, v in self.log.items()})
        print(f"log: {path}")


def analysis_thread(source, an, stop, stats=None):
    """Reads the ring one fast hop at a time; stats.an_lag gets, per hop, the time from the audio's delivery
    into the ring to the end of its analysis (queueing + compute)."""
    fh = an.fast_hop
    while not stop.is_set():
        r = source.ring.read(fh, timeout=0.5, stereo=True)
        if r is None:
            if source.done or source.ring.closed:
                an.mark_eof()
                return
            continue
        an.process(*r)
        if stats is not None:
            age = source.ring.age(an.samples)
            if age is not None:
                stats.an_lag.append(age)


def _style_twin_class():
    import cv2
    from dance_twin import Twin

    class StyleTwin(Twin):
        """dance_twin.Twin + the dance style in big letters and the genre line under it."""
        style_text = ""
        genre_text = ""
        pub_delay = None  # deque of s: start of a frame's render -> the frame is published (server side)

        def start_probe(self):
            from collections import deque
            self.pub_delay = deque(maxlen=2000)
            self._starts = {}
            threading.Thread(target=self._probe, daemon=True, name="twin-probe").start()

        def _probe(self):
            last = self.frame_id
            while not self._stop.is_set():
                fid = self.frame_id
                if fid != last:
                    t0 = self._starts.pop(fid, None)
                    if t0 is not None:
                        self.pub_delay.append(time.monotonic() - t0)
                    last = fid
                time.sleep(0.002)

        def _compose(self, *a, **k):
            if self.pub_delay is not None:
                self._starts[self.frame_id + 1] = time.monotonic()
                if len(self._starts) > 50:
                    self._starts.clear()
            img = super()._compose(*a, **k)
            font = cv2.FONT_HERSHEY_DUPLEX
            if self.style_text:
                txt = self.style_text
                (tw, th), _ = cv2.getTextSize(txt, font, 1.6, 3)
                x, y = (img.shape[1] - tw) // 2, 135
                cv2.rectangle(img, (x - 14, y - th - 14), (x + tw + 14, y + 14), (25, 25, 25), -1)
                cv2.putText(img, txt, (x, y), font, 1.6, (60, 220, 255), 3, cv2.LINE_AA)
            if self.genre_text:
                (tw, _), _ = cv2.getTextSize(self.genre_text, font, 0.6, 1)
                x = max(8, (img.shape[1] - tw) // 2)
                cv2.rectangle(img, (x - 8, 152), (x + tw + 8, 180), (25, 25, 25), -1)
                cv2.putText(img, self.genre_text, (x, 172), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
            return img

    return StyleTwin


def open_twin(args, start):
    if not args.view:
        return None
    try:
        Twin = _style_twin_class()
    except ImportError as e:
        print(f"view disabled: {e}")
        return None
    try:
        twin = Twin(port=args.view_port, start=start)
    except OSError as e:
        print(f"view disabled: port {args.view_port}: {e}")
        return None
    twin.start_probe()
    print(f"live view (left: MuJoCo sim, right: {'real arm' if args.real else 'target'}): {twin.url}")
    return twin


def measured_text(stats, source, twin, tick, ctl, an):
    """The measured part of the latency budget (analysis queue, tick lateness, twin pipeline, audio callbacks)."""
    out = ["measured latency (ms: mean / p99 / max):"]
    if stats.an_lag:
        a = np.array(stats.an_lag) * 1000.0
        out.append(f"  analysis (delivered -> analysed, {len(a)} hops) {a.mean():.1f} / {np.percentile(a, 99):.1f} / {a.max():.1f}")
    if stats.late:
        a = np.array(stats.late) * 1000.0
        out.append(f"  control tick lateness (+ {tick * 500:.0f} ms mean wait) {a.mean():.1f} / {np.percentile(a, 99):.1f} / {a.max():.1f}")
    if twin is not None and twin.pub_delay:
        a = np.array(twin.pub_delay) * 1000.0
        est = 500.0 / max(twin.status.get("fps", twin.fps), 1.0) + a.mean() + 5.0
        out.append(f"  twin: render start -> frame published {a.mean():.1f} / {np.percentile(a, 99):.1f} / {a.max():.1f};"
                   f" {twin.status.get('fps', 0):.0f} fps -> command visible on the stream ~{est:.0f} ms (+ browser)")
    cs = getattr(source, "callback_stats", lambda: None)()
    if cs:
        out.append(f"  audio callbacks: period {cs[0]:.2f} ms mean, {cs[1]:.1f} ms max"
                   + (f"; capture age of a block {cs[2]:.1f} ms (PortAudio timestamps)" if cs[2] is not None else "")
                   + f"; status errors {source.status_errors}")
    return "\n".join(out)


def vocals_wanted(args):
    return bool(getattr(args, "vocal_model", False))


def run_fast(args):
    """--file --fast: the whole pipeline on stream time, as fast as the CPU goes (no view, no sound)."""
    y = decode_file(args.file, 22050)
    ys = decode_stereo_like(args.file, 22050, len(y)) if vocals_wanted(args) else None
    an = make_analyzer(args, 22050)
    genre = open_genre(args)
    vocals = open_vocals(args)
    ctl = Controller(an, args, BASE, genre, vocals)
    end = len(y) / an.sr if args.duration is None else min(args.duration, len(y) / an.sr)
    tick, k, next_status = 1.0 / args.tick_hz, 0, 1.0
    t_ctl, t_genre, t_vox = 0.0, 1.0, 7.0
    fh = an.fast_hop
    while (k + 1) * fh <= len(y) and k * fh / an.sr < end:
        an.process(y[k * fh:(k + 1) * fh], None if ys is None else ys[:, k * fh:(k + 1) * fh])
        k += 1
        ts = k * fh / an.sr
        if genre is not None and ts >= t_genre:
            t_genre += genre.period
            if an.state.section != "silence":
                genre.classify(an.recent_audio(genre.window), an.sr, ts, an.rhythm(), an.feature_vector())
        if vocals is not None and ts >= t_vox:
            t_vox += vocals.period
            x = an.recent_audio_stereo(vocals.window)
            if x is not None and an.state.section != "silence":
                vocals.separate(x, an.sr, ts)
        while t_ctl + tick <= ts:
            t_ctl += tick
            q, st, info = ctl.tick(t_ctl, tick, t_ctl)
            if t_ctl >= next_status:
                if args.verbose or int(t_ctl) % 5 == 0:
                    print(ctl.status_line(t_ctl, st, info))
                next_status += 1.0
    print(f"done: {end:.0f} s of audio, peak commanded joint speed {ctl.vpeak:.2f} rad/s, "
          f"contacts {ctl.contacts}, limiter active {100 * ctl.clips / max(ctl.ticks, 1):.1f} % of ticks")
    if genre is not None and genre.costs:
        c = np.array(genre.costs) * 1000
        print(f"genre: {len(c)} classifications, {np.mean(c):.1f} ms mean, {np.max(c):.1f} ms max")
    if vocals is not None and vocals.costs:
        c = np.array(vocals.costs) * 1000
        print(f"vocals: {len(c)} separations of {vocals.window:.0f} s, {np.mean(c):.0f} ms mean, {np.max(c):.0f} ms max")
    return ctl


def run_live(args):
    """Real time: audio thread -> analysis thread -> 50 Hz control loop -> twin and/or real arm."""
    arm = start = None
    if args.real:
        arm = dance.Arm(args.host, args.port)
        if not arm.call({"cmd": "info"}).get("allow_motion", True):
            raise SystemExit("real_arm_server runs READ-ONLY: restart it with --allow-motion")
        start = arm.read()
        print("start pose: " + ", ".join(f"{j} {start[j]:+.2f}" for j in JOINTS))
    source = make_source(args)
    an = make_analyzer(args, source.sr, hop=source.hop)
    stats = RunStats()
    tick = 1.0 / args.tick_hz
    twin = open_twin(args, start=start if args.real else BASE)
    if twin is None and not args.real and not args.headless:
        raise SystemExit("--sim needs the view (mujoco + opencv), or --headless")
    ros = open_stream(args)
    stop = threading.Event()
    lagged = {"n": 0}
    styles = {"style": "", "genre": ""}

    def out(q, label):
        p = None
        if arm is not None:
            arm.send(q)
            p = arm.read()
        if twin is not None:
            twin.update(q, p, label)
        if ros is not None:
            ros.update(q, p, label, styles["style"], styles["genre"])
        return p

    try:
        if arm is not None:
            arm.call({"cmd": "torque", "on": True})
            print("going to the base pose...")
            dance.ramp(arm.read, BASE, args.approach_speed, lambda q: out(q, "approach: to the base pose"))
            t_end = time.monotonic() + args.settle_time  # keep commanding BASE until it is reached
            while True:
                p = out(BASE, "settling at the base pose")
                worst = max(abs(p[j] - BASE[j]) for j in JOINTS)
                if worst <= args.settle_tol:
                    break
                if time.monotonic() > t_end:
                    raise RuntimeError(f"base pose not reached: still {worst:.2f} rad away")
                time.sleep(0.02)
        elif twin is not None and args.view_wait > 0:
            print(f"open the page; starting in {args.view_wait:.0f} s...")
            t_end = time.monotonic() + args.view_wait
            while time.monotonic() < t_end:
                out(BASE, "base pose: waiting to start")
                time.sleep(0.02)

        genre = open_genre(args)
        vocals = open_vocals(args)
        ctl = Controller(an, args, BASE, genre, vocals)
        ctl.lead = sum(lead_terms(args, source.latency).values())
        bench = bench_analyzer(lambda: make_analyzer(args, source.sr, hop=source.hop), source.sr, source.hop)
        print_budget(budget_rows(args, an, source, tick, bench, twin))
        source.start()
        threading.Thread(target=analysis_thread, args=(source, an, stop, stats), daemon=True, name="analysis").start()
        if genre is not None:
            threading.Thread(target=genre.run, args=(an, stop), daemon=True, name="genre").start()
        if vocals is not None:
            threading.Thread(target=vocals.run, args=(an, stop), daemon=True, name="vocals").start()
        print(f"dancing live (Ctrl-C to stop); beat clock runs {ctl.lead * 1000:.0f} ms ahead")
        t0 = time.monotonic()
        next_tick, next_status, quiet_since = t0, t0 + 1.0, None
        next_budget = t0 + 8.0
        while True:
            now = time.monotonic()
            ts = source.stream_time(now)
            q, st, info = ctl.tick(ts, tick, now)
            if twin is not None:
                twin.style_text, twin.genre_text = ctl.style_text, ctl.genre_text
            styles["style"], styles["genre"] = ctl.style_text, ctl.genre_text
            p = out(q, ctl.label)
            late = time.monotonic() - next_tick
            ctl.late_max = max(ctl.late_max, late) if now - t0 > 2.0 else 0.0
            if now - t0 > 2.0:
                ctl.lates.append(late)
                stats.late.append(late)
            if next_budget is not None and now >= next_budget:
                next_budget = None
                print(measured_text(stats, source, twin, tick, ctl, an), flush=True)
            if p is not None:  # real arm: abort if it does not follow
                worst = max(abs(p[j] - q[j]) for j in JOINTS)
                lagged["n"] = lagged["n"] + 1 if worst > args.max_lag else 0
                if lagged["n"] >= 8:
                    raise RuntimeError(f"arm lags its target by {worst:.2f} rad (blocked?)")
            if now >= next_status:
                print(ctl.status_line(ts, st, info), flush=True)
                next_status += 1.0
            if args.duration is not None and ts >= args.duration:
                break
            if st.eof:  # file finished: back to BASE, then stop
                quiet_since = quiet_since or now
                if now - quiet_since > 3.0:
                    break
            next_tick += tick
            time.sleep(max(0.0, next_tick - time.monotonic()))
            if time.monotonic() - next_tick > 0.2:  # fell behind (machine busy): do not catch up in a burst
                next_tick = time.monotonic()
        stop.set()
        source.close()
        # leave smoothly: to BASE in sim, to the start pose on the real arm
        goal = {j: start[j] for j in JOINTS} if arm is not None else BASE
        if arm is not None:
            print("back to the start pose...")
            dance.ramp(arm.read, goal, args.approach_speed, lambda q: out(q, "return: to the start pose"))
            print("done: the arm holds its start pose (torque on). Hold it before stopping the server.")
        else:
            cur = dict(ctl.safety.q)
            dance.ramp(lambda: cur, goal, args.approach_speed, lambda q: out(q, "done"))
        print(measured_text(stats, source, twin, tick, ctl, an))
        print(f"peak commanded joint speed {ctl.vpeak:.2f} rad/s, contacts {ctl.contacts}, "
              f"audio overruns {source.ring.overruns}, control loop max late {ctl.late_max * 1000:.1f} ms"
              + (f" (p99 {np.percentile(ctl.lates, 99) * 1000:.1f} ms, {100 * np.mean(np.array(ctl.lates) > 0.025):.2f} % "
                 "of ticks > 25 ms late)" if ctl.lates else ""))
        if genre is not None and genre.costs:
            c = np.array(genre.costs) * 1000
            print(f"genre: {len(c)} classifications, {np.mean(c):.1f} ms mean, {np.max(c):.1f} ms max")
        if vocals is not None and vocals.costs:
            c = np.array(vocals.costs) * 1000
            print(f"vocals: {len(c)} separations, {np.mean(c):.0f} ms mean, {np.max(c):.0f} ms max")
        if twin is not None:
            dev = twin.max_deviation()
            print("twin: max |sim - " + ("real" if arm else "target") + "| per joint (rad): "
                  + ", ".join(f"{j} {v:.2f}" for j, v in dev.items()))
        return ctl
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\nSTOP ({str(e) or 'Ctrl-C'})" + (": holding the present pose" if arm is not None else ""))
        if arm is not None:
            arm.hold()
        if isinstance(e, RuntimeError):
            sys.exit(1)
    finally:
        stop.set()
        source.close()
        if twin is not None:
            twin.close()
        if ros is not None:
            ros.close()

