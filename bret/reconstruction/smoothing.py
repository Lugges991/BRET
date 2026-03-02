"""
Temporal smoothing for inferred percepts.

Encode categorical percepts as numeric values (house=1, face=0, mixed=0.5),
apply a 1-D filter, then re-threshold with optional hysteresis.
"""

import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ── encoding helpers ──────────────────────────────────────────────────────────

PERCEPT_TO_NUM = {"house": 1.0, "face": 0.0, "mixed": 0.5}
NUM_TO_PERCEPT = {1.0: "house", 0.0: "face", 0.5: "mixed"}


def encode_percepts(series: pd.Series) -> np.ndarray:
    """Map categorical percept labels → numeric (house=1, face=0, mixed=0.5)."""
    return series.map(PERCEPT_TO_NUM).to_numpy(dtype=float)


def decode_percepts(arr: np.ndarray, include_mixed: bool = True) -> np.ndarray:
    """Map numeric values back to categorical labels (no hysteresis)."""
    if include_mixed:
        return np.where(arr >= 0.75, "house", np.where(arr <= 0.25, "face", "mixed"))
    return np.where(arr >= 0.5, "house", "face")


# ── core filter + threshold ──────────────────────────────────────────────────

def smooth_and_threshold(
    arr: np.ndarray,
    *,
    method: str = "median",
    window_size: int = 150,
    threshold: float = 0.5,
    delta: float = 0.05,
    include_mixed: bool = True,
) -> np.ndarray:
    """Smooth a numeric percept array and re-threshold with hysteresis.

    Args:
        arr: 1-D float array (1=house, 0=face, 0.5=mixed).
        method: ``'mean'`` (uniform_filter1d) or ``'median'`` (median_filter).
        window_size: Filter window in samples (= ms at 1 kHz).
        threshold: Decision boundary (default 0.5).
        delta: Half-width of the hysteresis dead-band around *threshold*.
        include_mixed: If False, force binary classification (no mixed).

    Returns:
        1-D float array with values in {0, 0.5, 1}.
    """
    if method == "mean":
        smoothed = uniform_filter1d(arr.astype(float), size=window_size)
    elif method == "median":
        smoothed = median_filter(arr.astype(float), size=window_size)
    else:
        raise ValueError(f"Unknown smoothing method '{method}'. Use 'mean' or 'median'.")

    if include_mixed:
        result = np.where(
            smoothed >= threshold + delta,
            1.0,
            np.where(smoothed <= threshold - delta, 0.0, 0.5),
        )
    else:
        result = np.where(smoothed >= threshold, 1.0, 0.0)

    return result


# ── TemporalSmoother class ───────────────────────────────────────────────────

class TemporalSmoother:
    """
    Apply temporal smoothing to reduce noise in inferred percepts.

    Supports multiple smoothing methods and hyperparameter optimisation.
    """

    def __init__(self, include_mixed: bool = True):
        """Initialize smoother.

        Args:
            include_mixed: Whether the 3-class mixed state is allowed.
        """
        self.include_mixed = include_mixed
        self.optimal_params: Optional[Dict[str, float]] = None
        logger.info(f"Initialized TemporalSmoother (mixed={include_mixed})")

    # ── grid search ───────────────────────────────────────────────────────────

    def grid_search_optimal_window(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        method: str = "mean",
        window_range: Tuple[int, int] = (200, 2000),
        step_size: int = 10,
        threshold: float = 0.5,
        delta_candidates: list | None = None,
    ) -> Dict[str, float]:
        """Find optimal smoothing parameters via grid search.

        Searches over ``window_size × delta`` combinations and picks the pair
        that minimises mismatch (number of disagreements) between the smoothed
        prediction and ``y_true``.

        Args:
            y_true: Numeric encoded ground-truth array.
            y_pred: Numeric encoded prediction array (same length).
            method: Filter method ('mean' or 'median').
            window_range: (min, max) window sizes in samples.
            step_size: Step between candidate window sizes.
            threshold: Decision boundary.
            delta_candidates: List of delta values to try.

        Returns:
            Dict with keys ``window_size``, ``delta``, ``mismatch``.
        """
        if delta_candidates is None:
            delta_candidates = [0.02, 0.05, 0.07, 0.1]

        best_window: int = window_range[0]
        best_delta: float = delta_candidates[0]
        best_mismatch: float = float("inf")

        for ws in range(window_range[0], window_range[1] + step_size, step_size):
            for delta in delta_candidates:
                smoothed = smooth_and_threshold(
                    y_pred,
                    method=method,
                    window_size=ws,
                    threshold=threshold,
                    delta=delta,
                    include_mixed=self.include_mixed,
                )
                mismatch = int(np.sum(smoothed != y_true))
                if mismatch < best_mismatch:
                    best_mismatch = mismatch
                    best_window = ws
                    best_delta = delta

        self.optimal_params = {
            "window_size": best_window,
            "delta": best_delta,
            "mismatch": best_mismatch,
        }
        logger.info(
            f"Grid search result: window={best_window}, delta={best_delta}, "
            f"mismatch={best_mismatch}"
        )
        return self.optimal_params

    # ── filter wrappers ───────────────────────────────────────────────────────

    def apply_uniform_filter(
        self,
        df: pd.DataFrame,
        window_size: int,
        percept_col: str = "InferredPercept",
        threshold: float = 0.5,
        delta: float = 0.05,
    ) -> pd.DataFrame:
        """Apply uniform (moving average) filter to a percept column.

        Args:
            df: DataFrame containing ``percept_col``.
            window_size: Filter window in samples (= ms at 1 kHz).
            percept_col: Column name with categorical percept labels.
            threshold: Decision boundary.
            delta: Hysteresis half-width.

        Returns:
            DataFrame with added ``<percept_col>Smoothed`` column.
        """
        return self._apply_filter(
            df, "mean", window_size, percept_col, threshold, delta
        )

    def apply_median_filter(
        self,
        df: pd.DataFrame,
        window_size: int,
        percept_col: str = "InferredPercept",
        threshold: float = 0.5,
        delta: float = 0.05,
    ) -> pd.DataFrame:
        """Apply median filter to a percept column.

        Args:
            df: DataFrame containing ``percept_col``.
            window_size: Filter window in samples (= ms at 1 kHz).
            percept_col: Column name with categorical percept labels.
            threshold: Decision boundary.
            delta: Hysteresis half-width.

        Returns:
            DataFrame with added ``<percept_col>Smoothed`` column.
        """
        return self._apply_filter(
            df, "median", window_size, percept_col, threshold, delta
        )

    def apply_hysteresis_threshold(
        self,
        df: pd.DataFrame,
        delta: float,
        percept_col: str = "InferredPercept",
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Apply hysteresis thresholding (no smoothing filter).

        This is useful when the signal has already been filtered externally
        (e.g. by the Butterworth low-pass in preprocessing).

        Args:
            df: DataFrame with numeric percept values in ``percept_col``.
            delta: Hysteresis half-width.
            percept_col: Column with numeric (0/0.5/1) percept encoding.
            threshold: Decision boundary.

        Returns:
            DataFrame with added ``<percept_col>Thresholded`` column.
        """
        arr = df[percept_col].values.astype(float)

        if self.include_mixed:
            result = np.where(
                arr >= threshold + delta,
                1.0,
                np.where(arr <= threshold - delta, 0.0, 0.5),
            )
        else:
            result = np.where(arr >= threshold, 1.0, 0.0)

        out_col = f"{percept_col}Thresholded"
        df = df.copy()
        df[out_col] = result
        logger.info(
            f"Hysteresis threshold (δ={delta}) applied → '{out_col}'."
        )
        return df

    # ── private helper ────────────────────────────────────────────────────────

    def _apply_filter(
        self,
        df: pd.DataFrame,
        method: str,
        window_size: int,
        percept_col: str,
        threshold: float,
        delta: float,
    ) -> pd.DataFrame:
        if percept_col not in df.columns:
            raise ValueError(f"Column '{percept_col}' not found in DataFrame.")

        arr = encode_percepts(df[percept_col])
        smoothed = smooth_and_threshold(
            arr,
            method=method,
            window_size=window_size,
            threshold=threshold,
            delta=delta,
            include_mixed=self.include_mixed,
        )

        out_col = f"{percept_col}Smoothed"
        df = df.copy()
        # Store numeric value and categorical label
        df[out_col] = [NUM_TO_PERCEPT.get(v, "mixed") for v in smoothed]

        logger.info(
            f"{method.capitalize()} filter (window={window_size}, δ={delta}) "
            f"applied to '{percept_col}' → '{out_col}'."
        )
        return df
