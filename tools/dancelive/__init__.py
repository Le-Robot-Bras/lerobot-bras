"""dancelive - the real-time audio-reactive dance of tools/dance_live.py, split by concern.

    state     BeatState (analysis -> choreography interface), SECTIONS, SECTION_TIER, _ema
    audio_io  RingBuffer, DeviceSource, FileSource, decode_file, list_devices, make_source
    analysis  LiveAnalyzer: spectrum, onsets, bands, voice envelope, loudness, sections, DSP_FEATURES
    tempo     TempoTracker (mixin of LiveAnalyzer): tempo, half time, beat-clock PLL, confidence
    genre     YAMNet wrappers, genre / instrument -> style tables, GenreDetector, open_genre
    vocals    student (InstrumentDetector), Demucs VocalSeparator, vocal_gate, VocalFusion
    moves     beat envelopes, STYLE_MOVES, event poses, STYLES, derivative tables
    choreo    Dancer: combos, phrase / switch logic, events, accents, singer, mouth
    safety    budget_scale, SafetyFilter, ContactGuard, check_moves
    app       Controller (50 Hz tick + log), analysis_thread, twin view, run_fast, run_live
    cli       build_parser, main, dance_live (the Python API)

tools/dance_live.py re-exports all of it (python tools/dance_live.py ..., from dance_live import dance_live).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # tools/: dance, dance_music, dance_twin

from dance import BASE, JOINTS, LIMITS, MOVES, ROBOT_BODIES, SCENE, smooth01  # noqa: E402,F401
from dance_music import PHRASE, TIERS  # noqa: E402,F401  (no librosa needed: imported lazily there)

from .state import *  # noqa: E402,F401,F403
from .state import _ema  # noqa: E402,F401
from .audio_io import *  # noqa: E402,F401,F403
from .tempo import *  # noqa: E402,F401,F403
from .analysis import *  # noqa: E402,F401,F403
from .moves import *  # noqa: E402,F401,F403
from .moves import _derivs  # noqa: E402,F401
from .safety import *  # noqa: E402,F401,F403
from .choreo import *  # noqa: E402,F401,F403
from .vocals import *  # noqa: E402,F401,F403
from .genre import *  # noqa: E402,F401,F403
from .app import *  # noqa: E402,F401,F403
from .app import _style_twin_class  # noqa: E402,F401
from .cli import *  # noqa: E402,F401,F403
