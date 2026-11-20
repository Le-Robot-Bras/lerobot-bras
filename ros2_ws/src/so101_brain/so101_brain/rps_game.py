"""rps_game.py - Rock-Paper-Scissors game state machine for the SO-ARM101.

Pure Python (no ROS dependency) so it can be unit-tested independently and
driven at any control rate by hand_control_node.py.

Rules & robot presentation:
  - Triggered when a hand shows V_SIGN; tracks that specific hand's height.
  - COUNTDOWN: 3 beats ("Rock...", "Paper...", "Scissors...") bobbing at the
    tracked hand height with gripper closed and looking straight ahead.
  - REVEAL ("SHOOT!"): robot matches the player's hand height while showing:
      * SCISSORS -> gripper open (100 %), looking straight ahead
      * PAPER    -> gripper closed (0 %), looking straight ahead
      * ROCK     -> gripper closed (0 %), looking straight up
    Player's throw is read from the same hand:
      * CLOSED_HAND -> ROCK
      * OPEN_HAND   -> PAPER
      * V_SIGN      -> SCISSORS
  - Outcome reactions:
      * WIN  -> celebratory dance
      * LOSE -> retreat backward and look down ("sad")
      * DRAW -> quick side-to-side shrug
"""
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
import math

import numpy as np

ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


class RpsMove(Enum):
    """Valid Rock-Paper-Scissors throws."""

    ROCK = "ROCK"
    PAPER = "PAPER"
    SCISSORS = "SCISSORS"


class RpsOutcome(Enum):
    """Round result from the robot's perspective."""

    WIN = "WIN"
    LOSE = "LOSE"
    DRAW = "DRAW"


class RpsPhase(Enum):
    """Lifecycle phases of one Rock-Paper-Scissors round."""

    IDLE = auto()
    COUNTDOWN = auto()
    REVEAL = auto()
    HOLD_STANCE = auto()
    WIN_DANCE = auto()
    LOSE_SAD = auto()
    DRAW_REACTION = auto()
    FINISHED = auto()


GESTURE_TO_MOVE: dict[str, RpsMove] = {
    "CLOSED_HAND": RpsMove.ROCK,
    "OPEN_HAND": RpsMove.PAPER,
    "V_SIGN": RpsMove.SCISSORS,
}

_ROBOT_WINS_AGAINST: dict[RpsMove, RpsMove] = {
    RpsMove.ROCK: RpsMove.SCISSORS,
    RpsMove.PAPER: RpsMove.ROCK,
    RpsMove.SCISSORS: RpsMove.PAPER,
}

# Retreated and looking down (~57 deg downward pitch, min link z ~ 0.085 m in so101.urdf).
SAD_LOSE_POSE: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.95,
    "elbow_flex": 0.85,
    "wrist_flex": 1.10,
    "wrist_roll": 0.0,
}

# Celebratory dance keyframes: (arm_joints_rad, gripper_pct, hold_s).
# All joints are well inside URDF limits and links stay >= 0.22 m above the table.
WIN_DANCE_KEYFRAMES: tuple[tuple[dict[str, float], float, float], ...] = (
    (
        {
            "shoulder_pan": 0.60,
            "shoulder_lift": -0.25,
            "elbow_flex": -0.95,
            "wrist_flex": -0.35,
            "wrist_roll": 0.75,
        },
        100.0,
        0.65,
    ),
    (
        {
            "shoulder_pan": -0.60,
            "shoulder_lift": -0.05,
            "elbow_flex": -1.25,
            "wrist_flex": -0.20,
            "wrist_roll": -0.75,
        },
        15.0,
        0.70,
    ),
    (
        {
            "shoulder_pan": 0.55,
            "shoulder_lift": -0.25,
            "elbow_flex": -0.85,
            "wrist_flex": -0.40,
            "wrist_roll": 0.75,
        },
        100.0,
        0.65,
    ),
    (
        {
            "shoulder_pan": -0.55,
            "shoulder_lift": -0.05,
            "elbow_flex": -1.25,
            "wrist_flex": -0.15,
            "wrist_roll": -0.75,
        },
        15.0,
        0.70,
    ),
    (
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": -1.45,
            "wrist_flex": -0.15,
            "wrist_roll": 0.0,
        },
        100.0,
        0.55,
    ),
)

# Brief side-to-side shrug keyframes for a draw.
DRAW_KEYFRAMES: tuple[tuple[dict[str, float], float, float], ...] = (
    (
        {
            "shoulder_pan": 0.25,
            "shoulder_lift": -0.20,
            "elbow_flex": -0.40,
            "wrist_flex": 0.60,
            "wrist_roll": 0.50,
        },
        50.0,
        0.45,
    ),
    (
        {
            "shoulder_pan": -0.25,
            "shoulder_lift": -0.20,
            "elbow_flex": -0.40,
            "wrist_flex": 0.60,
            "wrist_roll": -0.50,
        },
        50.0,
        0.50,
    ),
    (
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.20,
            "elbow_flex": -0.40,
            "wrist_flex": 0.60,
            "wrist_roll": 0.0,
        },
        50.0,
        0.35,
    ),
)

_COUNTDOWN_CALLS = ("Rock...", "Paper...", "Scissors...")


@dataclass
class RpsConfig:
    """Timing, height bounds, and motion speeds for Rock-Paper-Scissors mode."""

    z_min: float = 0.12
    z_max: float = 0.35
    default_z: float = 0.22
    beat_duration_s: float = 1.20
    countdown_beats: int = 3
    bob_amplitude_m: float = 0.075
    countdown_wrist_nod_rad: float = 0.25
    reveal_settle_s: float = 1.0
    reveal_confirm_s: float = 0.35
    stance_hold_s: float = 1.5
    reveal_timeout_s: float = 5.0
    track_speed_rad_s: float = 2.0
    dance_speed_rad_s: float = 2.0
    sad_speed_rad_s: float = 0.55
    draw_speed_rad_s: float = 1.2
    sad_hold_s: float = 2.0
    z_smoothing_alpha: float = 0.35


@dataclass
class RpsCommand:
    """Target joint configuration and gripper opening emitted each control tick."""

    joints: dict[str, float]
    gripper_pct: float
    speed_rad_s: float


def decide_outcome(robot_move: RpsMove, player_move: RpsMove) -> RpsOutcome:
    """Determine the round outcome from the robot's perspective."""
    if robot_move == player_move:
        return RpsOutcome.DRAW
    if _ROBOT_WINS_AGAINST[robot_move] == player_move:
        return RpsOutcome.WIN
    return RpsOutcome.LOSE


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class RockPaperScissorsGame:
    """State machine managing a single Rock-Paper-Scissors round against one hand."""

    def __init__(
        self,
        config: RpsConfig | None = None,
        solve_height_pose: Callable[[float, bool], dict[str, float]] | None = None,
        log: Callable[[str], None] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self._config = config or RpsConfig()
        self._solve_height_pose = solve_height_pose or self._default_height_pose
        self._log = log or (lambda _msg: None)
        self._rng = rng if rng is not None else np.random.default_rng()

        self._phase = RpsPhase.IDLE
        self._target_hand_label: str = "Unknown"
        self._target_z: float = self._clamp_z(self._config.default_z)
        self._has_z_sample: bool = False
        self._phase_start_time: float = 0.0
        self._last_logged_beat: int = -1

        self._robot_move: RpsMove | None = None
        self._pending_outcome: RpsOutcome | None = None
        self._player_gesture: str | None = None
        self._confirm_gesture: str | None = None
        self._confirm_since: float = 0.0

        self._keyframe_idx: int = 0
        self._keyframe_start_time: float = 0.0
        self._last_commanded_joints: dict[str, float] = {n: 0.0 for n in ARM_JOINT_NAMES}
        self._sad_total_duration_s: float = self._config.sad_hold_s

    @property
    def phase(self) -> RpsPhase:
        return self._phase

    @property
    def target_hand_label(self) -> str:
        return self._target_hand_label

    @property
    def target_z(self) -> float:
        return self._target_z

    @property
    def robot_move(self) -> RpsMove | None:
        return self._robot_move

    def start(self, hand_label: str, now: float, initial_z: float | None = None) -> None:
        """Start a new Rock-Paper-Scissors round against `hand_label`."""
        self._target_hand_label = hand_label or "Unknown"
        self._phase = RpsPhase.COUNTDOWN
        self._phase_start_time = now
        self._last_logged_beat = -1

        if initial_z is not None and math.isfinite(initial_z):
            self._target_z = self._clamp_z(initial_z)
            self._has_z_sample = True
        else:
            self._target_z = self._clamp_z(self._config.default_z)
            self._has_z_sample = False

        self._robot_move = None
        self._pending_outcome = None
        self._player_gesture = None
        self._confirm_gesture = None
        self._confirm_since = 0.0
        self._keyframe_idx = 0
        self._keyframe_start_time = now
        self._sad_total_duration_s = self._config.sad_hold_s
        self._log(
            f"RPS: V_SIGN detected on {self._target_hand_label} hand — starting round!"
        )

    def reset(self) -> None:
        """Return the game state machine to IDLE."""
        self._phase = RpsPhase.IDLE
        self._robot_move = None
        self._pending_outcome = None
        self._player_gesture = None
        self._confirm_gesture = None
        self._confirm_since = 0.0
        self._last_logged_beat = -1

    def update_hands(
        self,
        hands: list[dict],
        primary_hand_pose: np.ndarray | None,
        primary_pose_fresh: bool,
        now: float,
    ) -> None:
        """Update tracked hand height and gesture from /hand/fingers_state."""
        if self._phase in (RpsPhase.IDLE, RpsPhase.FINISHED):
            return

        matched = self._find_target_hand(hands)
        if matched is None:
            self._player_gesture = None
            return

        idx, hand_data = matched
        raw_z: float | None = None
        if idx == 0 and primary_pose_fresh and primary_hand_pose is not None:
            raw_z = float(primary_hand_pose[2])
        else:
            palm_norm = hand_data.get("palm_norm")
            if isinstance(palm_norm, (list, tuple)) and len(palm_norm) >= 2:
                ny = float(palm_norm[1])
                raw_z = self._config.z_max - ny * (self._config.z_max - self._config.z_min)

        if raw_z is not None and math.isfinite(raw_z):
            clamped_z = self._clamp_z(raw_z)
            if not self._has_z_sample:
                self._target_z = clamped_z
                self._has_z_sample = True
            else:
                a = _clamp(self._config.z_smoothing_alpha, 0.01, 1.0)
                self._target_z = a * clamped_z + (1.0 - a) * self._target_z

        gesture = hand_data.get("gesture")
        self._player_gesture = str(gesture) if gesture is not None else None

    def step(self, now: float) -> RpsCommand | None:
        """Advance the round by one control tick and return the active command."""
        if self._phase in (RpsPhase.IDLE, RpsPhase.FINISHED):
            return None

        if self._phase == RpsPhase.COUNTDOWN:
            cmd = self._step_countdown(now)
        elif self._phase == RpsPhase.REVEAL:
            cmd = self._step_reveal(now)
        elif self._phase == RpsPhase.HOLD_STANCE:
            cmd = self._step_hold_stance(now)
        elif self._phase == RpsPhase.WIN_DANCE:
            cmd = self._step_keyframes(now, WIN_DANCE_KEYFRAMES, self._config.dance_speed_rad_s)
        elif self._phase == RpsPhase.DRAW_REACTION:
            cmd = self._step_keyframes(now, DRAW_KEYFRAMES, self._config.draw_speed_rad_s)
        elif self._phase == RpsPhase.LOSE_SAD:
            cmd = self._step_lose_sad(now)
        else:
            cmd = None

        if cmd is not None:
            self._last_commanded_joints = dict(cmd.joints)
        return cmd

    # ------------------------------------------------------------------ phase steps

    def _step_countdown(self, now: float) -> RpsCommand | None:
        beat_dur = max(self._config.beat_duration_s, 0.05)
        total_cd = beat_dur * self._config.countdown_beats
        elapsed = max(0.0, now - self._phase_start_time)

        if elapsed < total_cd:
            beat_idx = min(int(elapsed / beat_dur), self._config.countdown_beats - 1)
            if beat_idx != self._last_logged_beat:
                self._last_logged_beat = beat_idx
                call = (
                    _COUNTDOWN_CALLS[beat_idx]
                    if beat_idx < len(_COUNTDOWN_CALLS)
                    else f"Beat {beat_idx + 1}..."
                )
                self._log(f"RPS: {call}")

            beat_phase = (elapsed % beat_dur) / beat_dur
            sine = math.sin(2.0 * math.pi * beat_phase)
            bob_z = self._clamp_z(self._target_z + self._config.bob_amplitude_m * sine)
            joints = self._solve_height_pose(bob_z, False)
            joints["wrist_flex"] = _clamp(
                joints["wrist_flex"] - self._config.countdown_wrist_nod_rad * sine,
                -1.60,
                1.60,
            )
            return RpsCommand(
                joints=joints,
                gripper_pct=0.0,
                speed_rad_s=self._config.track_speed_rad_s,
            )

        # Countdown finished -> pick a random throw and reveal immediately.
        moves = list(RpsMove)
        self._robot_move = moves[int(self._rng.integers(0, len(moves)))]
        self._phase = RpsPhase.REVEAL
        self._phase_start_time = now
        self._confirm_gesture = None
        self._confirm_since = 0.0
        self._log(f"RPS: SHOOT! Robot plays {self._robot_move.name}!")
        return self._step_reveal(now)

    def _robot_throw_command(self) -> RpsCommand:
        """Build the joint + gripper command for the robot's chosen throw."""
        assert self._robot_move is not None
        look_up = self._robot_move == RpsMove.ROCK
        gripper_pct = 100.0 if self._robot_move == RpsMove.SCISSORS else 0.0
        return RpsCommand(
            joints=self._solve_height_pose(self._target_z, look_up),
            gripper_pct=gripper_pct,
            speed_rad_s=self._config.track_speed_rad_s,
        )

    def _step_reveal(self, now: float) -> RpsCommand | None:
        assert self._robot_move is not None
        reveal_elapsed = max(0.0, now - self._phase_start_time)

        if reveal_elapsed >= self._config.reveal_settle_s:
            if self._player_gesture in GESTURE_TO_MOVE:
                if self._confirm_gesture != self._player_gesture:
                    self._confirm_gesture = self._player_gesture
                    self._confirm_since = now
                elif (now - self._confirm_since) >= self._config.reveal_confirm_s:
                    player_move = GESTURE_TO_MOVE[self._confirm_gesture]
                    outcome = decide_outcome(self._robot_move, player_move)
                    self._log(
                        f"RPS: Robot={self._robot_move.name} vs "
                        f"{self._target_hand_label}={player_move.name} -> {outcome.name}!"
                    )
                    self._pending_outcome = outcome
                    self._phase = RpsPhase.HOLD_STANCE
                    self._phase_start_time = now
                    return self._robot_throw_command()
            else:
                self._confirm_gesture = None

        if reveal_elapsed >= self._config.reveal_timeout_s:
            self._log("RPS: Timed out waiting for player's throw; round ended.")
            self._phase = RpsPhase.FINISHED
            return None

        return self._robot_throw_command()

    def _step_hold_stance(self, now: float) -> RpsCommand | None:
        """Hold the revealed throw stance briefly so the robot's choice is clearly visible."""
        if (now - self._phase_start_time) >= self._config.stance_hold_s:
            outcome = self._pending_outcome
            self._pending_outcome = None
            if outcome is None:
                self._phase = RpsPhase.FINISHED
                return None
            return self._enter_outcome_phase(outcome, now)
        return self._robot_throw_command()

    def _enter_outcome_phase(self, outcome: RpsOutcome, now: float) -> RpsCommand | None:
        if outcome == RpsOutcome.WIN:
            self._phase = RpsPhase.WIN_DANCE
            self._keyframe_idx = 0
            self._keyframe_start_time = now
            self._log("RPS: Robot won! Doing celebratory dance.")
            return self._step_keyframes(now, WIN_DANCE_KEYFRAMES, self._config.dance_speed_rad_s)

        if outcome == RpsOutcome.LOSE:
            self._phase = RpsPhase.LOSE_SAD
            self._phase_start_time = now
            max_delta = max(
                abs(SAD_LOSE_POSE[n] - self._last_commanded_joints.get(n, 0.0))
                for n in ARM_JOINT_NAMES
            )
            travel_s = max_delta / max(self._config.sad_speed_rad_s, 1e-3)
            self._sad_total_duration_s = travel_s + self._config.sad_hold_s
            self._log("RPS: Robot lost... retreating and looking down.")
            return self._step_lose_sad(now)

        self._phase = RpsPhase.DRAW_REACTION
        self._keyframe_idx = 0
        self._keyframe_start_time = now
        self._log("RPS: Draw!")
        return self._step_keyframes(now, DRAW_KEYFRAMES, self._config.draw_speed_rad_s)

    def _step_keyframes(
        self,
        now: float,
        keyframes: tuple[tuple[dict[str, float], float, float], ...],
        speed_rad_s: float,
    ) -> RpsCommand | None:
        while self._keyframe_idx < len(keyframes):
            joints, gripper_pct, hold_s = keyframes[self._keyframe_idx]
            if (now - self._keyframe_start_time) < hold_s:
                return RpsCommand(
                    joints=dict(joints),
                    gripper_pct=gripper_pct,
                    speed_rad_s=speed_rad_s,
                )
            self._keyframe_idx += 1
            self._keyframe_start_time = now

        self._phase = RpsPhase.FINISHED
        self._log("RPS: Round complete.")
        return None

    def _step_lose_sad(self, now: float) -> RpsCommand | None:
        if (now - self._phase_start_time) >= self._sad_total_duration_s:
            self._phase = RpsPhase.FINISHED
            self._log("RPS: Round complete.")
            return None
        return RpsCommand(
            joints=dict(SAD_LOSE_POSE),
            gripper_pct=0.0,
            speed_rad_s=self._config.sad_speed_rad_s,
        )

    # ------------------------------------------------------------------ helpers

    def _find_target_hand(self, hands: list[dict]) -> tuple[int, dict] | None:
        if not hands:
            return None
        if self._target_hand_label == "Unknown":
            first = hands[0]
            return (0, first) if isinstance(first, dict) else None
        for idx, hand_data in enumerate(hands):
            if isinstance(hand_data, dict) and hand_data.get("hand") == self._target_hand_label:
                return idx, hand_data
        return None

    def _clamp_z(self, z: float) -> float:
        z_lo = min(self._config.z_min, self._config.z_max)
        z_hi = max(self._config.z_min, self._config.z_max)
        return _clamp(float(z), z_lo, z_hi)

    def _default_height_pose(self, target_z: float, look_up: bool) -> dict[str, float]:
        """Lightweight planar fallback mapping target_z to arm joints without Pinocchio."""
        span = max(self._config.z_max - self._config.z_min, 1e-6)
        t = _clamp((target_z - self._config.z_min) / span, 0.0, 1.0)
        sl = -0.55 - 0.15 * math.sin(math.pi * t)
        ef = 1.15 - 1.70 * t
        pitch = -math.pi / 2.0 if look_up else 0.0
        wf = _clamp(pitch - sl - ef, -1.60, 1.60)
        return {
            "shoulder_pan": 0.0,
            "shoulder_lift": sl,
            "elbow_flex": ef,
            "wrist_flex": wf,
            "wrist_roll": 0.0,
        }
