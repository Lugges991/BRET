"""
Visualization module for eye tracking data.
"""

from bret.visualization.gaze_plots import plot_gaze_trajectory, plot_gaze_heatmap
from bret.visualization.percept_plots import plot_percept_timeline
from bret.visualization.quality_plots import (
    plot_preprocessing_quality,
    plot_subject_quality_heatmap,
    plot_blink_rate_comparison,
    plot_calibration_quality,
    plot_quality_vs_accuracy,
)

__all__ = [
    "plot_gaze_trajectory",
    "plot_gaze_heatmap",
    "plot_percept_timeline",
    "plot_preprocessing_quality",
    "plot_subject_quality_heatmap",
    "plot_blink_rate_comparison",
    "plot_calibration_quality",
    "plot_quality_vs_accuracy",
]
