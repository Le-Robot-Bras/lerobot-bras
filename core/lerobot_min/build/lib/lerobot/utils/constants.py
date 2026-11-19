# Slimmed-down version of lerobot/utils/constants.py (Apache-2.0, HuggingFace Inc.)
# Stripped of the dataset/policy/HF Hub constants. Calibration stored locally.
import os
from pathlib import Path

OBS_STR = "observation"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGE = OBS_STR + ".image"
OBS_IMAGES = OBS_IMAGE + "s"
ACTION = "action"

ROBOTS = "robots"
TELEOPERATORS = "teleoperators"

# Calibration: ~/.cache/tek5_robotics/calibration (can be overridden by env var)
_default_home = Path.home() / ".cache" / "tek5_robotics"
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", _default_home)).expanduser()
HF_LEROBOT_CALIBRATION = Path(
    os.getenv("HF_LEROBOT_CALIBRATION", HF_LEROBOT_HOME / "calibration")
).expanduser()
