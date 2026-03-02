"""
Preprocessing module for EyeLink eye tracking data.

This module handles raw .asc file parsing, data cleaning, filtering,
alignment, and quality validation.
"""

from bret.preprocessing.parser import parse_asc_file, read_eyelink_data
from bret.preprocessing.cleaning import (
    interpolate_blinks,
    detect_outliers,
    coalesce_events,
)
from bret.preprocessing.filtering import apply_butterworth_filter
from bret.preprocessing.alignment import align_to_center, calculate_centroid
from bret.preprocessing.validation import (
    exclude_low_quality_trials,
    check_missing_data_threshold,
    extract_trial_data,
)
from bret.preprocessing.pipeline import PreprocessingPipeline

__all__ = [
    "parse_asc_file",
    "interpolate_blinks",
    "detect_outliers",
    "coalesce_events",
    "apply_butterworth_filter",
    "align_to_center",
    "calculate_centroid",
    "exclude_low_quality_trials",
    "check_missing_data_threshold",
    "PreprocessingPipeline",
    "extract_trial_data",
]
