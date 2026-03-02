"""
Temporal feature calculation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def compute_fixation_duration(
    df: pd.DataFrame,
    timestamp_column: str = "Timestamp",
) -> pd.DataFrame:
    """
    Compute elapsed time since current fixation started.
    
    Args:
        df: DataFrame with fixation events
        timestamp_column: Name of timestamp column
        
    Returns:
        DataFrame with 'fixation_elapsed_time' column
    """
    # TODO: Implement from mk_features.py
    raise NotImplementedError("Fixation duration not yet implemented")


def compute_rolling_majority(
    df: pd.DataFrame,
    column: str,
    window_size: float = 0.5,
    sampling_rate: float = 250.0,
) -> pd.DataFrame:
    """
    Compute rolling majority vote within time window.
    
    Args:
        df: DataFrame with percept column
        column: Column name to aggregate
        window_size: Window size in seconds
        sampling_rate: Sampling rate in Hz
        
    Returns:
        DataFrame with rolling majority column
    """
    # TODO: Implement from mk_features.py rolling_majority()
    # Calculate rolling mode over 500ms window
    
    raise NotImplementedError("Rolling majority not yet implemented")


def label_percept_transitions(
    df: pd.DataFrame,
    jump_window_ms: float = 200.0,
    smoothing_window_ms: float = 400.0,
    sampling_rate: float = 250.0,
    percept_col: str = "InferredPercept",
) -> pd.DataFrame:
    """Label each percept transition as jump-induced or spontaneous.

    Applies a per-trial median smooth to ``percept_col`` to suppress noise
    flips, then detects rows where the smoothed percept changes.  Each
    transition is classified as ``'jump_induced'`` if any fixpoint jump
    (``FixpointSection`` increment) occurred within ±``jump_window_ms`` in the
    same trial, otherwise as ``'spontaneous'``.

    At 250 Hz each sample is 4 ms, so ``smoothing_window_ms`` is converted to
    samples as ``round(smoothing_window_ms / (1000 / sampling_rate))``.

    Args:
        df: Percepts DataFrame with at minimum ``Timestamp``, ``Trial``,
            ``FixpointSection``, and ``percept_col`` columns.
        jump_window_ms: Half-width (ms) of the window around a fixpoint jump
            within which a percept switch is classified as jump-induced.
        smoothing_window_ms: Median-filter window applied per trial before
            detecting transitions.  Should be wide enough to remove noise
            flips (≥200 ms) but narrower than typical dominance durations.
        sampling_rate: Data sampling rate in Hz (default 250).
        percept_col: Source percept column (sample-level, unsmoothed).

    Returns:
        DataFrame with three new columns:
        - ``InferredPerceptSmoothed``: per-trial smoothed percept labels.
        - ``PerceptTransition``: bool, True on the first sample of a new
          percept epoch (NaN/mixed boundaries are skipped).
        - ``TransitionType``: ``'jump_induced'``, ``'spontaneous'``, or
          ``None`` for non-transition rows.
    """
    from bret.reconstruction.smoothing import smooth_and_threshold, encode_percepts, decode_percepts

    ms_per_sample = 1000.0 / sampling_rate
    window_samples = max(1, round(smoothing_window_ms / ms_per_sample))
    # Ensure odd window for median filter symmetry
    if window_samples % 2 == 0:
        window_samples += 1

    result = df.copy()
    result["InferredPerceptSmoothed"] = pd.NA
    result["PerceptTransition"] = False
    result["TransitionType"] = None

    for trial_id, grp in result.groupby("Trial", sort=False):
        idx = grp.index

        # --- smooth per trial ---
        encoded = encode_percepts(grp[percept_col])
        smoothed_numeric = smooth_and_threshold(
            encoded,
            method="median",
            window_size=window_samples,
            threshold=0.5,
            delta=0.05,
            include_mixed=True,
        )
        smoothed_labels = decode_percepts(smoothed_numeric, include_mixed=True)
        result.loc[idx, "InferredPerceptSmoothed"] = smoothed_labels

        # --- fixpoint jump timestamps (FixpointSection increments) ---
        fp_section = grp["FixpointSection"]
        jump_mask = fp_section.diff().fillna(0) != 0
        jump_timestamps = grp.loc[jump_mask, "Timestamp"].to_numpy(dtype=float)

        # --- detect percept transitions on smoothed signal ---
        smooth_series = pd.Series(smoothed_labels, index=idx)
        prev = smooth_series.shift(1)
        # A transition: percept changes AND neither value is NaN
        trans_mask = (smooth_series != prev) & prev.notna() & smooth_series.notna()
        trans_mask.iloc[0] = False  # first sample is never a "switch"

        # Timestamps are in seconds; convert window to matching units
        jump_window_s = jump_window_ms / 1000.0

        for row_idx in trans_mask[trans_mask].index:
            t = result.at[row_idx, "Timestamp"]
            result.at[row_idx, "PerceptTransition"] = True
            if len(jump_timestamps) > 0 and np.any(np.abs(jump_timestamps - t) <= jump_window_s):
                result.at[row_idx, "TransitionType"] = "jump_induced"
            else:
                result.at[row_idx, "TransitionType"] = "spontaneous"

    logger.info(
        f"label_percept_transitions: {result['PerceptTransition'].sum()} transitions detected "
        f"({(result['TransitionType'] == 'jump_induced').sum()} jump-induced, "
        f"{(result['TransitionType'] == 'spontaneous').sum()} spontaneous) "
        f"across {result['Trial'].nunique()} trials "
        f"[smooth={smoothing_window_ms}ms={window_samples}samp, jump_window=±{jump_window_ms}ms]"
    )
    return result


def summarize_transitions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Summarise jump-induced vs spontaneous transitions per trial and for the run.

    Expects ``df`` to contain ``PerceptTransition`` and ``TransitionType``
    columns as produced by :func:`label_percept_transitions`.

    Args:
        df: DataFrame with transition labels.

    Returns:
        Tuple of:
        - per_trial_df: DataFrame with columns ``Trial``, ``n_transitions``,
          ``n_jump_induced``, ``n_spontaneous``, ``pct_jump_induced``.
        - run_summary: Dict with run-level aggregates of the same fields.
    """
    transitions = df[df["PerceptTransition"] == True].copy()

    per_trial = (
        transitions.groupby("Trial")
        .agg(
            n_transitions=("PerceptTransition", "sum"),
            n_jump_induced=("TransitionType", lambda s: (s == "jump_induced").sum()),
            n_spontaneous=("TransitionType", lambda s: (s == "spontaneous").sum()),
        )
        .reset_index()
    )
    per_trial["pct_jump_induced"] = (
        per_trial["n_jump_induced"] / per_trial["n_transitions"].replace(0, np.nan) * 100
    ).round(1)

    total_t = int(per_trial["n_transitions"].sum())
    total_ji = int(per_trial["n_jump_induced"].sum())
    total_sp = int(per_trial["n_spontaneous"].sum())
    run_summary: Dict = {
        "n_trials": int(per_trial["Trial"].nunique()),
        "n_transitions": total_t,
        "n_jump_induced": total_ji,
        "n_spontaneous": total_sp,
        "pct_jump_induced": round(total_ji / total_t * 100, 1) if total_t > 0 else float("nan"),
    }

    logger.info(
        f"summarize_transitions: {total_t} total transitions — "
        f"{total_ji} jump-induced ({run_summary['pct_jump_induced']}%), "
        f"{total_sp} spontaneous"
    )
    return per_trial, run_summary
