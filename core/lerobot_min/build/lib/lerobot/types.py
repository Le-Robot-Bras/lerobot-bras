# Slimmed-down version of lerobot/types.py (Apache-2.0, HuggingFace Inc.)
# Stripped of the torch/policy types the Tek5 module does not need.
from typing import Any

RobotAction = dict[str, Any]
RobotObservation = dict[str, Any]
