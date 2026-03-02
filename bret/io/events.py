"""
Events TSV generation for BIDS-compatible output.

Converts per-run percept data into epoch-collapsed events.tsv files.

Data flow
---------
Report runs:
    sub-{N}_run{RR}_report_perceptData_*.csv   (ground-truth button-press epochs)
    + percepts/{stem}_percepts.csv              (for jump detection when requested)
    → events/{stem}_events.tsv

No-report runs:
    percepts/{stem}_percepts.csv                (InferredPercept, sample-level)
    → epoch-collapse → events/{stem}_events.tsv

Timing note
-----------
Onsets are computed from scanner timing, not from eye-tracker timestamps:

    trial_offset(N) = n_dummies * tr + iti + (iti + trial_dur) * (N - 1)

Where N is the 1-based trial number.  ``Timestamp`` (trial-relative seconds)
is added on top of the trial offset for each epoch's within-trial onset.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── timing helpers ────────────────────────────────────────────────────────────

def compute_trial_offsets(
    percepts_df: pd.DataFrame,
    n_dummies: int = 5,
    tr: float = 1.75,
    trial_dur: float = 120.0,
    iti: float = 20.0,
) -> Dict[float, float]:
    """Return scanner-absolute start time (seconds) for each trial.

    Uses the same formula as the BIDS conversion script::

        trial_offset(N) = n_dummies * tr + iti + (iti + trial_dur) * (N - 1)

    where N is the 1-based trial number.

    Args:
        percepts_df: Percepts sidecar DataFrame with a ``Trial`` column.
        n_dummies: Number of dummy TRs discarded before the run.
        tr: Repetition time (s).
        trial_dur: Expected duration of each trial (s).
        iti: Inter-trial interval (s).

    Returns:
        Dict mapping trial id → scanner-absolute start time (s).
    """
    trial_ids = sorted(percepts_df["Trial"].dropna().unique())
    offsets: Dict[float, float] = {}
    for n, trial_id in enumerate(trial_ids, start=1):
        offsets[float(trial_id)] = n_dummies * tr + iti + (iti + trial_dur) * (n - 1)
    return offsets


# ── jump-induced suppression ──────────────────────────────────────────────────

def suppress_jump_induced_epochs(
    percepts_df: pd.DataFrame,
    percept_col: str,
    jump_window_ms: float = 200.0,
    smoothing_window_ms: float = 400.0,
    sampling_rate: float = 250.0,
) -> pd.Series:
    """Return a cleaned percept Series with jump-induced epochs deleted.

    For each trial, any epoch that was *entered* via a jump-induced transition
    is suppressed: samples within that epoch are replaced with the preceding
    epoch's percept value, effectively absorbing the jump-induced epoch into
    its predecessor.

    Uses :func:`~bret.features.temporal.label_percept_transitions` internally
    to identify jump-induced transitions and their boundaries.

    Args:
        percepts_df: Percepts sidecar DataFrame.  Must contain ``Timestamp``,
            ``Trial``, ``FixpointSection``, and ``percept_col``.
        percept_col: Column name to clean (e.g. ``'InferredPercept'`` or
            ``'ReportedPercept'``).
        jump_window_ms: Half-window (ms) for jump-induced classification.
        smoothing_window_ms: Median-filter window (ms) applied before
            transition detection.
        sampling_rate: Sampling rate in Hz.

    Returns:
        pd.Series (same index as ``percepts_df``) with jump-induced epochs
        replaced.
    """
    from bret.features.temporal import label_percept_transitions

    logger.info(
        f"Suppressing jump-induced epochs in '{percept_col}' "
        f"(jump_window=±{jump_window_ms}ms, smooth={smoothing_window_ms}ms)"
    )

    # label transitions on the chosen column
    labeled = label_percept_transitions(
        percepts_df,
        jump_window_ms=jump_window_ms,
        smoothing_window_ms=smoothing_window_ms,
        sampling_rate=sampling_rate,
        percept_col=percept_col,
    )

    # if the col was smoothed, use the smoothed version as working series
    work_col = (
        "InferredPerceptSmoothed"
        if percept_col == "InferredPercept" and "InferredPerceptSmoothed" in labeled.columns
        else percept_col
    )

    result = labeled[work_col].copy()

    for _trial_id, grp in labeled.groupby("Trial", sort=False):
        idx = list(grp.index)
        vals = result.loc[idx].tolist()
        trans_type = labeled.loc[idx, "TransitionType"].tolist()

        pos = 0
        while pos < len(idx):
            if trans_type[pos] == "jump_induced":
                pre_val = vals[pos - 1] if pos > 0 else vals[pos]
                # find end of this jump-induced epoch:
                # the next row that is a *spontaneous* transition or end-of-trial
                end = pos + 1
                while end < len(idx):
                    if trans_type[end] == "spontaneous":
                        break
                    end += 1
                # forward-fill pre_val through the jump-induced epoch
                for k in range(pos, end):
                    vals[k] = pre_val
                    trans_type[k] = None  # mark as processed
                result.loc[idx[pos:end]] = vals[pos:end]
                pos = end
            else:
                pos += 1

    n_suppressed = (labeled["TransitionType"] == "jump_induced").sum()
    logger.info(f"Suppressed {n_suppressed} jump-induced transitions")
    return result


# ── epoch collapse ────────────────────────────────────────────────────────────

def collapse_to_epochs(
    percepts_df: pd.DataFrame,
    percept_series: pd.Series,
    trial_offsets: Dict[float, float],
    run_type_prefix: str,
) -> pd.DataFrame:
    """Collapse a sample-level percept series into epochs with onset/duration.

    Each contiguous run of the same percept label within a trial becomes one
    epoch row.  ``onset`` = ``trial_offsets[trial_id]`` + trial-relative
    ``Timestamp`` of the epoch start.

    Args:
        percepts_df: Percepts sidecar DataFrame.  Must contain ``Trial`` and
            ``Timestamp`` (trial-relative seconds).
        percept_series: Sample-level percept labels (same index as
            ``percepts_df``).
        trial_offsets: Dict of trial_id → scanner-absolute start time (s).
        run_type_prefix: Prefix for ``trial_type``, e.g. ``'no_report'``.
            Produces labels like ``'no_report_rivalry_face'``.

    Returns:
        DataFrame with columns: ``onset``, ``duration``, ``trial_type``.
    """
    rows: List[dict] = []

    for trial_id, grp in percepts_df.groupby("Trial", sort=False):
        grp = grp.sort_values("Timestamp")
        trial_percepts = percept_series.loc[grp.index].reset_index(drop=True)
        timestamps = grp["Timestamp"].to_numpy(dtype=float)
        trial_start = trial_offsets.get(float(trial_id), 0.0)

        # trial type label (rivalry / replay) from the data
        if "Type" in grp.columns and not grp["Type"].dropna().empty:
            type_label = str(grp["Type"].dropna().iloc[0]).lower()
        else:
            type_label = "rivalry"

        epoch_start_idx = 0
        current_percept = trial_percepts.iloc[0]

        for i in range(1, len(trial_percepts)):
            p = trial_percepts.iloc[i]
            if p != current_percept:
                rows.append({
                    "onset": round(trial_start + timestamps[epoch_start_idx], 1),
                    "duration": round(timestamps[i] - timestamps[epoch_start_idx], 1),
                    "trial_type": f"{run_type_prefix}_{type_label}_{current_percept}",
                })
                epoch_start_idx = i
                current_percept = p

        # final epoch
        rows.append({
            "onset": round(trial_start + timestamps[epoch_start_idx], 1),
            "duration": round(timestamps[-1] - timestamps[epoch_start_idx], 1),
            "trial_type": f"{run_type_prefix}_{type_label}_{current_percept}",
        })

    df = pd.DataFrame(rows, columns=["onset", "duration", "trial_type"])
    return df.sort_values("onset").reset_index(drop=True)


# ── epoch merging ────────────────────────────────────────────────────────────

def merge_short_epochs(
    df: pd.DataFrame,
    min_duration: float = 1.75,
) -> pd.DataFrame:
    """Merge epochs shorter than *min_duration* into a neighbouring epoch.

    Epochs are only merged with neighbours that are **contiguous** (no gap,
    i.e. within the same trial).  An epoch with a gap on both sides — an
    isolated single-percept trial — is left unchanged.

    Merging strategy:
    - Short epoch that has a contiguous **predecessor** → absorbed into it
      (predecessor duration extended).
    - Short epoch with no predecessor but a contiguous **successor** → onset
      of successor is shifted back and its duration extended.

    Repeated until no epoch shorter than *min_duration* remains.

    Args:
        df: Events DataFrame with ``onset``, ``duration``, ``trial_type``.
        min_duration: Minimum acceptable epoch duration (s).  Defaults to TR
            (1.75 s).

    Returns:
        Filtered DataFrame (copy), sorted by onset.
    """
    if df.empty:
        return df.copy()

    df = df.copy().reset_index(drop=True)
    tol = 0.15  # seconds — rounding tolerance for contiguity check

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(df):
            if df.loc[i, "duration"] < min_duration:
                merged = False
                # try predecessor
                if i > 0:
                    pred_end = df.loc[i - 1, "onset"] + df.loc[i - 1, "duration"]
                    if abs(pred_end - df.loc[i, "onset"]) <= tol:
                        df.loc[i - 1, "duration"] = round(
                            df.loc[i - 1, "duration"] + df.loc[i, "duration"], 1
                        )
                        df = df.drop(i).reset_index(drop=True)
                        changed = True
                        merged = True
                # try successor
                if not merged and i < len(df) - 1:
                    cur_end = df.loc[i, "onset"] + df.loc[i, "duration"]
                    if abs(cur_end - df.loc[i + 1, "onset"]) <= tol:
                        df.loc[i + 1, "onset"] = df.loc[i, "onset"]
                        df.loc[i + 1, "duration"] = round(
                            df.loc[i + 1, "duration"] + df.loc[i, "duration"], 1
                        )
                        df = df.drop(i).reset_index(drop=True)
                        changed = True
                        merged = True
                if not merged:
                    # isolated short epoch (e.g. only epoch in trial) — leave it
                    i += 1
            else:
                i += 1

    return df.reset_index(drop=True)


# ── main builders ─────────────────────────────────────────────────────────────

def build_events_from_report(
    percept_data: pd.DataFrame,
    trial_offsets: Dict[float, float],
    percepts_df: Optional[pd.DataFrame] = None,
    exclude_jump_induced: bool = False,
    jump_window_ms: float = 200.0,
    smoothing_window_ms: float = 400.0,
    sampling_rate: float = 250.0,
) -> pd.DataFrame:
    """Build events DataFrame from ground-truth button-press perceptData.

    Uses the MATLAB-generated ``*_report_perceptData_*.csv`` as the source of
    percept epochs (columns: ``trial``, ``type``, ``leftImage``, ``percept``,
    ``onset``, ``duration``).  ``onset`` values are trial-relative; scanner
    absolute onset = ``trial_offsets[trial_id] + epoch_onset``.

    ``trial_type`` is formatted as ``report_{type}_{percept}``.

    When ``exclude_jump_induced=True``, epochs whose onset falls within
    ±``jump_window_ms`` of a fixation jump are dropped and their duration is
    merged into the preceding epoch.

    Args:
        percept_data: Ground-truth epoch DataFrame.
        trial_offsets: Dict of trial_id → scanner-absolute start time (s).
        percepts_df: Percepts sidecar (required when ``exclude_jump_induced``).
        exclude_jump_induced: Drop epochs entered via fixation jumps.
        jump_window_ms: Half-window (ms) for jump classification.
        smoothing_window_ms: Smoothing window for transition detection.
        sampling_rate: Sampling rate in Hz.

    Returns:
        DataFrame with columns: ``onset``, ``duration``, ``trial_type``.
    """
    jump_ts_by_trial: Dict[float, np.ndarray] = {}
    jump_window_s = jump_window_ms / 1000.0
    if exclude_jump_induced and percepts_df is not None:
        for trial_id, grp in percepts_df.groupby("Trial", sort=False):
            fp = grp["FixpointSection"]
            jump_mask = fp.diff().fillna(0) != 0
            jump_ts_by_trial[float(trial_id)] = grp.loc[jump_mask, "Timestamp"].to_numpy(dtype=float)

    rows: List[dict] = []
    for _, epoch in percept_data.iterrows():
        trial_id = float(epoch["trial"])
        trial_start = trial_offsets.get(trial_id, 0.0)
        onset_abs = trial_start + float(epoch["onset"])
        duration = float(epoch["duration"])
        percept = str(epoch["percept"])
        type_label = str(epoch["type"]).lower()

        if exclude_jump_induced and trial_id in jump_ts_by_trial:
            trial_relative_onset = float(epoch["onset"])
            jts = jump_ts_by_trial[trial_id]
            if len(jts) > 0 and np.any(np.abs(jts - trial_relative_onset) <= jump_window_s):
                logger.debug(
                    f"Dropping jump-induced epoch: trial={trial_id} "
                    f"onset={trial_relative_onset:.3f}s percept={percept}"
                )
                if rows:
                    rows[-1]["duration"] = round(rows[-1]["duration"] + duration, 1)
                continue

        rows.append({
            "onset": round(onset_abs, 1),
            "duration": round(duration, 1),
            "trial_type": f"report_{type_label}_{percept}",
        })

    df = pd.DataFrame(rows, columns=["onset", "duration", "trial_type"])
    return df.sort_values("onset").reset_index(drop=True)


def build_events_from_noreport(
    percepts_df: pd.DataFrame,
    trial_offsets: Dict[float, float],
    exclude_jump_induced: bool = False,
    jump_window_ms: float = 200.0,
    smoothing_window_ms: float = 400.0,
    sampling_rate: float = 250.0,
) -> pd.DataFrame:
    """Build events DataFrame from inferred percepts (no-report runs).

    ``trial_type`` is formatted as ``no_report_{type}_{percept}``.

    Args:
        percepts_df: Percepts sidecar DataFrame.
        trial_offsets: Dict of trial_id → scanner-absolute start time (s).
        exclude_jump_induced: Suppress jump-induced epochs.
        jump_window_ms: Half-window (ms) for jump classification.
        smoothing_window_ms: Pre-smoothing window (ms).
        sampling_rate: Sampling rate in Hz.

    Returns:
        DataFrame with columns: ``onset``, ``duration``, ``trial_type``.
    """
    # Use section-level aggregated percept (already one label per fixation section)
    # rather than the raw sample-level InferredPercept column.  This avoids
    # thousands of sub-TR micro-epochs that make no sense for fMRI analysis.
    if "InferredPerceptAggregated" in percepts_df.columns:
        base_col = "InferredPerceptAggregated"
    else:
        base_col = "InferredPercept"

    if exclude_jump_induced:
        percept_series = suppress_jump_induced_epochs(
            percepts_df,
            percept_col=base_col,
            jump_window_ms=jump_window_ms,
            smoothing_window_ms=smoothing_window_ms,
            sampling_rate=sampling_rate,
        )
    else:
        percept_series = percepts_df[base_col].copy()

    return collapse_to_epochs(
        percepts_df,
        percept_series,
        trial_offsets=trial_offsets,
        run_type_prefix="no_report",
    )


# ── top-level entry point ─────────────────────────────────────────────────────

def build_events_tsv(
    subject_dir: Path,
    stem: str,
    run_type: str,
    exclude_jump_induced: bool = False,
    jump_window_ms: float = 200.0,
    smoothing_window_ms: float = 400.0,
    sampling_rate: float = 250.0,
    n_dummies: int = 5,
    tr: float = 1.75,
    trial_dur: float = 120.0,
    iti: float = 20.0,
    min_epoch_duration: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build events and switch-events DataFrames for one run.

    After building raw epochs, any epoch shorter than ``min_epoch_duration``
    is merged into its contiguous neighbour (same trial).  This prevents
    sub-TR stutter epochs that would be unresolvable in the fMRI signal.

    Args:
        subject_dir: Subject directory (e.g. ``data/sub-11``).
        stem: Run file stem, e.g. ``'s11r06r'``.
        run_type: ``'report'`` or ``'no-report'``.
        exclude_jump_induced: Drop epochs entered via fixation jumps.
        jump_window_ms: Half-window (ms) for jump classification.
        smoothing_window_ms: Pre-smoothing window (ms).
        sampling_rate: Sampling rate in Hz.
        n_dummies: Number of dummy TRs discarded before the run.
        tr: Repetition time (s).
        trial_dur: Expected duration of each trial (s).
        iti: Inter-trial interval (s).
        min_epoch_duration: Minimum epoch duration (s).  Epochs shorter than
            this are merged into a contiguous neighbour.  Defaults to ``tr``.

    Returns:
        Tuple ``(events_df, switch_df)`` where ``switch_df`` is identical
        but with all durations set to 0 (instantaneous switch events).

    Raises:
        FileNotFoundError: When required input files are missing.
    """
    if min_epoch_duration is None:
        min_epoch_duration = tr
    from bret.io.loaders import load_percept_reports

    subject_dir = Path(subject_dir)
    percepts_path = subject_dir / "percepts" / f"{stem}_percepts.csv"
    if not percepts_path.exists():
        raise FileNotFoundError(f"Percepts sidecar not found: {percepts_path}")

    percepts_df = pd.read_csv(percepts_path)
    trial_offsets = compute_trial_offsets(percepts_df, n_dummies=n_dummies, tr=tr,
                                          trial_dur=trial_dur, iti=iti)

    logger.info(
        f"Building events for {stem} ({run_type}), "
        f"{len(trial_offsets)} trials "
        f"[n_dummies={n_dummies} TR={tr}s trial_dur={trial_dur}s ITI={iti}s], "
        f"exclude_jump_induced={exclude_jump_induced}"
    )

    if run_type == "report":
        run_match = re.search(r"r(\d+)r$", stem)
        if not run_match:
            raise ValueError(f"Cannot parse run number from report stem '{stem}'")
        run_nr = int(run_match.group(1))
        percept_data = load_percept_reports(subject_dir, run_nr)
        percept_data = percept_data.copy()
        percept_data["trial"] = percept_data["trial"].astype(float)
        events_df = build_events_from_report(
            percept_data=percept_data,
            trial_offsets=trial_offsets,
            percepts_df=percepts_df if exclude_jump_induced else None,
            exclude_jump_induced=exclude_jump_induced,
            jump_window_ms=jump_window_ms,
            smoothing_window_ms=smoothing_window_ms,
            sampling_rate=sampling_rate,
        )
    else:
        events_df = build_events_from_noreport(
            percepts_df=percepts_df,
            trial_offsets=trial_offsets,
            exclude_jump_induced=exclude_jump_induced,
            jump_window_ms=jump_window_ms,
            smoothing_window_ms=smoothing_window_ms,
            sampling_rate=sampling_rate,
        )

    n_raw = len(events_df)
    events_df = merge_short_epochs(events_df, min_duration=min_epoch_duration)
    n_merged = n_raw - len(events_df)
    if n_merged:
        logger.info(
            f"Merged {n_merged} sub-{min_epoch_duration:.2f}s epochs into neighbours "
            f"({n_raw} → {len(events_df)})"
        )

    switch_df = events_df.copy()
    switch_df["duration"] = 0.0

    logger.info(f"Built {len(events_df)} event epochs for {stem}")
    return events_df, switch_df


# ── writer ────────────────────────────────────────────────────────────────────

def save_events_tsv(
    events_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write events DataFrame to a tab-separated .tsv file.

    Args:
        events_df: DataFrame with ``onset``, ``duration``, ``trial_type``.
        output_path: Destination path (parent directories created if needed).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved events ({len(events_df)} epochs) → {output_path}")
