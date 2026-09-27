"""Quota-weighted brief sampling with hardware/leaf compatibility and steering.

Callers import ``plan_briefs``, ``leaf_hardware_allowed``, and ``load_angles``.
Hardware labels stay compatible with the leaf claim. This package does
not write questions or review them.
"""

from cuda_sft_qgen.planner.sampling import (
    _instance_hint,
    _pick_angle,
    leaf_hardware_allowed,
    load_angles,
    plan_briefs,
)
from cuda_sft_qgen.planner.joint import LeafCoverage

__all__ = [
    "plan_briefs",
    "leaf_hardware_allowed",
    "load_angles",
    "_instance_hint",
    "_pick_angle",
    "LeafCoverage",
]
