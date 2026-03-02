"""
Feature engineering module for eye tracking data.
"""

from bret.features.spatial import calculate_distance_to_fixation
from bret.features.temporal import (
    compute_fixation_duration,
    compute_rolling_majority,
)
from bret.features.motion import compute_velocity, compute_acceleration

__all__ = [
    "calculate_distance_to_fixation",
    "compute_fixation_duration",
    "compute_rolling_majority",
    "compute_velocity",
    "compute_acceleration",
]
