"""
Data quality check functions for eye tracking data.

Computes per-trial and per-run quality metrics from preprocessed DataFrames,
including missing data percentages, blink/saccade rates, gaze dispersion,
and fixation quality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def validate_asc_file(filepath: Path) -> Dict[str, Any]:
    """
    Validate .asc file structure and content.

    Checks that the file exists, is readable, contains sample lines,
    MSG lines with trial metadata, and the required GAZE_COORDS header.

    Args:
        filepath: Path to .asc file

    Returns:
        Dictionary with validation results including:
        - valid (bool): overall pass/fail
        - errors (list[str]): descriptions of failures
        - n_sample_lines, n_msg_lines, has_gaze_coords, has_trials
    """
    filepath = Path(filepath)
    result: Dict[str, Any] = {
        "valid": True,
        "errors": [],
        "n_sample_lines": 0,
        "n_msg_lines": 0,
        "has_gaze_coords": False,
        "has_trials": False,
    }

    if not filepath.exists():
        result["valid"] = False
        result["errors"].append(f"File does not exist: {filepath}")
        return result

    try:
        with open(filepath, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Cannot read file: {e}")
        return result

    if len(lines) < 10:
        result["valid"] = False
        result["errors"].append(f"File has only {len(lines)} lines — likely truncated")
        return result

    for line in lines:
        stripped = line.strip()
        if stripped and stripped[0].isdigit():
            result["n_sample_lines"] += 1
        elif stripped.startswith("MSG"):
            result["n_msg_lines"] += 1
            if "Trial #" in stripped:
                result["has_trials"] = True
        elif "GAZE_COORDS" in stripped:
            result["has_gaze_coords"] = True

    if result["n_sample_lines"] == 0:
        result["valid"] = False
        result["errors"].append("No sample data lines found")
    if not result["has_gaze_coords"]:
        result["valid"] = False
        result["errors"].append("Missing GAZE_COORDS header")
    if not result["has_trials"]:
        result["errors"].append("No trial metadata MSG lines found (may be non-rivalry file)")

    logger.info(
        f"Validated {filepath.name}: {result['n_sample_lines']} samples, "
        f"{result['n_msg_lines']} messages, valid={result['valid']}"
    )
    return result


def compute_trial_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-trial quality metrics from a preprocessed DataFrame.

    Args:
        df: Preprocessed DataFrame with columns Timestamp, X, Y, Diameter,
            Trial, Fixation, Saccade, Blink, DistanceToFixpoint1,
            DistanceToFixpoint2, DistanceToCenter.

    Returns:
        DataFrame indexed by Trial with quality metrics:
        - n_samples: sample count
        - duration_s: trial duration in seconds
        - blink_pct: percentage of blink samples
        - saccade_pct: percentage of saccade samples
        - fixation_pct: percentage of fixation samples
        - gaze_std_x, gaze_std_y: gaze position dispersion (standard deviation)
        - mean_dist_center: mean distance to screen center (degrees)
        - median_diameter: median pupil diameter
        - diameter_cv: coefficient of variation of pupil diameter
    """
    required = {"Timestamp", "X", "Y", "Diameter", "Trial", "Fixation", "Saccade", "Blink"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    records = []
    for trial_id, tdf in df.groupby("Trial"):
        n = len(tdf)
        ts = tdf["Timestamp"].values
        duration = (ts[-1] - ts[0]) if n > 1 else 0.0

        blink_pct = (tdf["Blink"] == 1).mean() * 100
        saccade_pct = (tdf["Saccade"] == 1).mean() * 100
        fixation_pct = (tdf["Fixation"] == 1).mean() * 100

        gaze_std_x = tdf["X"].std()
        gaze_std_y = tdf["Y"].std()

        mean_dist_center = (
            tdf["DistanceToCenter"].mean() if "DistanceToCenter" in tdf.columns else np.nan
        )

        diam = tdf["Diameter"]
        median_diam = diam.median()
        diameter_cv = diam.std() / diam.mean() if diam.mean() != 0 else np.nan

        records.append(
            {
                "Trial": trial_id,
                "n_samples": n,
                "duration_s": duration,
                "blink_pct": blink_pct,
                "saccade_pct": saccade_pct,
                "fixation_pct": fixation_pct,
                "gaze_std_x": gaze_std_x,
                "gaze_std_y": gaze_std_y,
                "mean_dist_center": mean_dist_center,
                "median_diameter": median_diam,
                "diameter_cv": diameter_cv,
            }
        )

    result = pd.DataFrame(records)
    logger.info(f"Computed trial-level quality for {len(result)} trials")
    return result


def compute_run_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute aggregate quality metrics for an entire run (all trials combined).

    Args:
        df: Preprocessed DataFrame for one run.

    Returns:
        Dictionary with aggregate quality metrics:
        - n_samples, n_trials, total_duration_s
        - blink_pct, saccade_pct, fixation_pct (overall)
        - mean_gaze_std_x, mean_gaze_std_y
        - mean_dist_center
        - median_diameter, diameter_cv
        - trial_quality: per-trial DataFrame (from compute_trial_quality)
    """
    trial_q = compute_trial_quality(df)

    n_samples = len(df)
    n_trials = int(df["Trial"].nunique())
    blink_pct = (df["Blink"] == 1).mean() * 100
    saccade_pct = (df["Saccade"] == 1).mean() * 100
    fixation_pct = (df["Fixation"] == 1).mean() * 100

    gaze_std_x = df["X"].std()
    gaze_std_y = df["Y"].std()
    mean_dist_center = (
        df["DistanceToCenter"].mean() if "DistanceToCenter" in df.columns else np.nan
    )

    diam = df["Diameter"]
    median_diam = float(diam.median())
    diameter_cv = float(diam.std() / diam.mean()) if diam.mean() != 0 else np.nan

    ts = df["Timestamp"].values
    total_dur = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0

    result = {
        "n_samples": n_samples,
        "n_trials": n_trials,
        "total_duration_s": total_dur,
        "blink_pct": float(blink_pct),
        "saccade_pct": float(saccade_pct),
        "fixation_pct": float(fixation_pct),
        "gaze_std_x": float(gaze_std_x),
        "gaze_std_y": float(gaze_std_y),
        "mean_dist_center": float(mean_dist_center),
        "median_diameter": median_diam,
        "diameter_cv": diameter_cv,
        "trial_quality": trial_q,
    }
    logger.info(
        f"Run quality: {n_samples} samples, {n_trials} trials, "
        f"blink={blink_pct:.1f}%, saccade={saccade_pct:.1f}%"
    )
    return result


def check_calibration_quality(df: pd.DataFrame) -> Dict[str, float]:
    """
    Assess gaze data quality as a proxy for calibration accuracy.

    Computes how tightly gaze clusters around fixation points during fixation
    events (lower = better calibration).

    Args:
        df: Preprocessed DataFrame with distance columns.

    Returns:
        Dictionary with:
        - mean_min_fixpoint_dist: mean of min(dist1, dist2) during fixations
        - median_min_fixpoint_dist: median of the same
        - pct_within_1deg: % of fixation samples within 1° of nearest fixpoint
        - pct_within_2deg: % of fixation samples within 2° of nearest fixpoint
    """
    dist_cols = ["DistanceToFixpoint1", "DistanceToFixpoint2"]
    if not all(c in df.columns for c in dist_cols):
        logger.warning("Distance columns not found — cannot assess calibration quality")
        return {}

    fix_mask = df["Fixation"] == 1
    if fix_mask.sum() == 0:
        logger.warning("No fixation samples found")
        return {}

    fix_df = df.loc[fix_mask]
    min_dist = fix_df[dist_cols].min(axis=1)

    result = {
        "mean_min_fixpoint_dist": float(min_dist.mean()),
        "median_min_fixpoint_dist": float(min_dist.median()),
        "pct_within_1deg": float((min_dist <= 1.0).mean() * 100),
        "pct_within_2deg": float((min_dist <= 2.0).mean() * 100),
    }
    logger.info(
        f"Calibration quality: median min dist = {result['median_min_fixpoint_dist']:.2f}°, "
        f"{result['pct_within_1deg']:.1f}% within 1°"
    )
    return result


def detect_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect anomalies in eye tracking data.

    Flags include:
    - Trials with excessive blink percentage (>30%)
    - Velocity spikes (gaze jumps > 50°/s between consecutive samples)
    - Pupil diameter outliers (beyond median ± 4σ)

    Args:
        df: Preprocessed DataFrame.

    Returns:
        Dictionary with:
        - high_blink_trials: list of trial IDs with >30% blinks
        - velocity_spike_count: number of inter-sample jumps > threshold
        - diameter_outlier_count: number of diameter outlier samples
        - anomaly_summary: human-readable summary string
    """
    result: Dict[str, Any] = {
        "high_blink_trials": [],
        "velocity_spike_count": 0,
        "diameter_outlier_count": 0,
        "anomaly_summary": "",
    }

    # High blink trials
    for trial_id, tdf in df.groupby("Trial"):
        blink_pct = (tdf["Blink"] == 1).mean()
        if blink_pct > 0.3:
            result["high_blink_trials"].append(int(trial_id))

    # Velocity spikes (simple inter-sample displacement in pixels)
    dx = df["X"].diff()
    dy = df["Y"].diff()
    displacement = np.sqrt(dx**2 + dy**2)
    # at 1000Hz, 1 sample = 1ms; 50 pixels/sample ≈ huge jump
    spike_threshold = 100  # pixels per sample
    result["velocity_spike_count"] = int((displacement > spike_threshold).sum())

    # Diameter outliers
    diam = df["Diameter"]
    med = diam.median()
    std = diam.std()
    outliers = ((diam < med - 4 * std) | (diam > med + 4 * std)) & diam.notna()
    result["diameter_outlier_count"] = int(outliers.sum())

    issues = []
    if result["high_blink_trials"]:
        issues.append(f"{len(result['high_blink_trials'])} trials with >30% blinks")
    if result["velocity_spike_count"] > 100:
        issues.append(f"{result['velocity_spike_count']} velocity spikes")
    if result["diameter_outlier_count"] > 50:
        issues.append(f"{result['diameter_outlier_count']} diameter outliers")

    result["anomaly_summary"] = "; ".join(issues) if issues else "No major anomalies"
    logger.info(f"Anomaly detection: {result['anomaly_summary']}")
    return result


def compute_subject_quality_report(
    subject_dir: Path,
) -> pd.DataFrame:
    """
    Compute quality metrics for all preprocessed runs of a subject.

    Args:
        subject_dir: Path to subject directory (e.g. data/sub-11/).

    Returns:
        DataFrame with one row per run, columns:
        - subject, run, run_type, filename
        - n_samples, n_trials, total_duration_s
        - blink_pct, saccade_pct, fixation_pct
        - gaze_std_x, gaze_std_y, mean_dist_center
        - median_diameter, diameter_cv
        - mean_min_fixpoint_dist, pct_within_1deg, pct_within_2deg
        - anomaly_summary
    """
    import re

    subject_dir = Path(subject_dir)
    preprocessed_dir = subject_dir / "preprocessed"
    if not preprocessed_dir.exists():
        logger.warning(f"No preprocessed/ dir in {subject_dir}")
        return pd.DataFrame()

    files = sorted(preprocessed_dir.glob("*_preprocessed.csv"))
    if not files:
        logger.warning(f"No preprocessed files in {preprocessed_dir}")
        return pd.DataFrame()

    sub_match = re.search(r"sub-(\d+)", subject_dir.name)
    sub_nr = int(sub_match.group(1)) if sub_match else 0

    records = []
    for fpath in files:
        stem = fpath.stem  # e.g. s11r04r_preprocessed
        run_match = re.search(r"r(\d+)(nr|r)", stem)
        run_nr = int(run_match.group(1)) if run_match else 0
        run_type = "no-report" if "nr" in stem else "report"

        logger.info(f"Computing quality for {fpath.name}")
        df = pd.read_csv(fpath)

        run_q = compute_run_quality(df)
        cal_q = check_calibration_quality(df)
        anom = detect_anomalies(df)

        row = {
            "subject": sub_nr,
            "run": run_nr,
            "run_type": run_type,
            "filename": fpath.name,
            "n_samples": run_q["n_samples"],
            "n_trials": run_q["n_trials"],
            "total_duration_s": run_q["total_duration_s"],
            "blink_pct": run_q["blink_pct"],
            "saccade_pct": run_q["saccade_pct"],
            "fixation_pct": run_q["fixation_pct"],
            "gaze_std_x": run_q["gaze_std_x"],
            "gaze_std_y": run_q["gaze_std_y"],
            "mean_dist_center": run_q["mean_dist_center"],
            "median_diameter": run_q["median_diameter"],
            "diameter_cv": run_q["diameter_cv"],
            **cal_q,
            "anomaly_summary": anom["anomaly_summary"],
            "high_blink_trials": str(anom["high_blink_trials"]),
            "velocity_spikes": anom["velocity_spike_count"],
        }
        records.append(row)

    report = pd.DataFrame(records)
    logger.info(f"Quality report for sub-{sub_nr}: {len(report)} runs")
    return report
