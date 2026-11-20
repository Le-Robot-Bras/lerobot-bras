"""Shared vocabulary between the modules: the analysis -> choreography interface (BeatState), the
section names, and the EMA coefficient helper."""
import math
from dataclasses import dataclass
from typing import Optional

SECTIONS = ["silence", "calm", "normal", "drop"]
SECTION_TIER = {"calm": 0, "normal": 1, "drop": 2}

@dataclass
class BeatState:
    """Snapshot of the analysis. The beat clock: musical beat theta(t) = theta_ref + (t - t_ref) * rate,
    dance beat = (theta - off) / div (div = 2 when the song is danced half time)."""
    t: float = 0.0          # stream time of the latest analysed frame (s)
    t_ref: float = 0.0
    theta_ref: float = 0.0
    rate: float = 1.5       # musical beats per second
    off: float = 0.0
    div: int = 1
    music_bpm: float = 0.0  # 0 = no tempo yet
    bpm: float = 0.0        # danced bpm
    conf: float = 0.0       # 0..1
    locked: bool = False
    bass: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    loud_db: float = -120.0
    energy: float = 0.0     # 0..1, loudness relative to the song's loud parts
    section: str = "silence"
    vox_env: float = 0.0    # 0..1 fast (~50 ms) envelope of the harmonic energy in the voice band
    eof: bool = False
    # --- latency / anticipation fields (added; the fields above are read by choreo / vocals / genre)
    t_audio: float = 0.0    # stream time up to which the audio has been analysed (s)
    loud_fast_db: float = -120.0  # ~0.1 s loudness (dB): the fast twin of loud_db (3 s)
    jump_db: float = 0.0    # loud_fast_db minus the level ~1 s earlier: > +4 dB = a hit, < -6 dB = a drop-out
    energy_fast: float = 0.0  # 0..1 rank of loud_fast_db in the song's own distribution (no 3 s lag)
    build_level: float = 0.0  # 0..1 how much the music looks like a build-up (rising energy / risers / snare roll)
    drop_eta: Optional[float] = None  # seconds (stream time) until the predicted drop, None = none expected
    drop_t: Optional[float] = None    # the predicted drop as a stream time (s): drop_eta = drop_t - now
    drop_theta: Optional[float] = None  # the predicted drop as a musical beat number (theta); a drop lands on a bar line

    def beat(self, t):
        return (self.theta_ref + (t - self.t_ref) * self.rate - self.off) / self.div

    def dance_beat_of(self, theta):
        """The dance beat (as beat() returns it) of a musical beat number theta, e.g. drop_theta."""
        return (theta - self.off) / self.div


def _ema(dt, tau):
    return 1.0 - math.exp(-dt / tau)
