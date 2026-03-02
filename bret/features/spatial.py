"""
Spatial feature calculation.
"""

from typing import Tuple


import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_distance_to_fixation(
    data: pd.DataFrame,
    fixation_spots: np.ndarray,
    center: Tuple[float, float],
    deg_per_pix_width: float,
) -> pd.DataFrame:
    """Calculate distances (degrees) from gaze to fixation spots and center.

    Args:
        data: DataFrame with required columns: X, Y, Fixpoint1, Fixpoint2
        fixation_spots: Array of shape (N, 2) with 1-based fixpoint ids
        center: (x, y) coordinates for stimulus center
        deg_per_pix_width: degrees per pixel (x-axis)

    Returns:
        DataFrame with added distance columns in degrees.
    """
    logger.info("Calculating distances to fixation spots and center.")

    required = {"X", "Y", "Fixpoint1", "Fixpoint2"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = data.copy()

    # Backfill fixation indices within sections (fixation jumps)
    df["Fixpoint1"] = df["Fixpoint1"].bfill()
    df["Fixpoint2"] = df["Fixpoint2"].bfill()

    fix_spots = np.asarray(fixation_spots, dtype=float)
    if fix_spots.ndim != 2 or fix_spots.shape[1] != 2:
        raise ValueError("fixation_spots must have shape (N, 2)")

    fix1_idx = df["Fixpoint1"].to_numpy(dtype=float)
    fix2_idx = df["Fixpoint2"].to_numpy(dtype=float)

    fix1 = np.full((len(df), 2), np.nan, dtype=float)
    fix2 = np.full((len(df), 2), np.nan, dtype=float)

    valid1 = (~pd.isna(fix1_idx)) & (fix1_idx >= 1) & (fix1_idx <= fix_spots.shape[0])
    valid2 = (~pd.isna(fix2_idx)) & (fix2_idx >= 1) & (fix2_idx <= fix_spots.shape[0])

    fix1[valid1] = fix_spots[fix1_idx[valid1].astype(int) - 1]
    fix2[valid2] = fix_spots[fix2_idx[valid2].astype(int) - 1]

    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()

    dist_fix1 = np.hypot(x - fix1[:, 0], y - fix1[:, 1]) * deg_per_pix_width
    dist_fix2 = np.hypot(x - fix2[:, 0], y - fix2[:, 1]) * deg_per_pix_width
    dist_center = np.hypot(x - center[0], y - center[1]) * deg_per_pix_width

    df["DistanceToFixpoint1"] = dist_fix1
    df["DistanceToFixpoint2"] = dist_fix2
    df["DistanceToCenter"] = dist_center

    return df