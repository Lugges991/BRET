"""
Data saving functions.
"""

import json
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_preprocessed_data(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save preprocessed data to CSV.

    Args:
        df: Preprocessed DataFrame (output of PreprocessingPipeline.process_file)
        output_path: Destination .csv path; parent directories are created if needed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved preprocessed data ({len(df)} rows) → {output_path}")


# Columns written to the slim percept sidecar.
# Timestamp + Trial are the join keys; the rest are derived outputs.
_SIDECAR_COLS = [
    "Timestamp",
    "Trial",
    "Type",  # rivalry vs replay — needed for events TSV labelling
    "InferredPercept",
    "InferredPerceptMixed",
    "FixpointSection",
    "InferredPerceptAggregated",
    "ReportedPercept",  # only present on report runs
]


def save_percept_data(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save a slim percept sidecar CSV alongside the preprocessed file.

    Only the reconstruction-derived columns are written — ``Timestamp`` and
    ``Trial`` as join keys, plus ``InferredPercept``, ``InferredPerceptMixed``,
    ``FixpointSection``, ``InferredPerceptAggregated``, and ``ReportedPercept``
    (the last only when present, i.e. on report runs).  All raw gaze columns
    remain in the preprocessed CSV, avoiding duplication.

    To get the full dataset at analysis time::

        from bret.io.loaders import load_percept_sidecar
        df = load_percept_sidecar(percepts_path, preprocessed_path)

    Args:
        df: Full reconstruction DataFrame (output of ``run_reconstruct``).
        output_path: Destination .csv path; parent directories are created.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [c for c in _SIDECAR_COLS if c in df.columns]
    out = df[cols]
    out.to_csv(output_path, index=False)
    logger.info(f"Saved percept sidecar ({len(out)} rows, {len(cols)} cols) → {output_path}")


def save_evaluation_results(
    results: dict,
    quality_dir: Path,
    stem: str,
) -> None:
    """
    Persist evaluation results from :meth:`ReconstructionEvaluator.evaluate_run`.

    Writes three files under *quality_dir*:
    - ``{stem}_eval_by_section.csv`` — one row per fixpoint section
    - ``{stem}_eval_by_trial.csv``   — one row per trial
    - ``{stem}_eval_summary.json``   — flat summary metrics

    Args:
        results: Dict returned by ``ReconstructionEvaluator.evaluate_run()``.
        quality_dir: Output directory (created if needed).
        stem: File stem, e.g. ``'s12r03r'``.
    """
    quality_dir = Path(quality_dir)
    quality_dir.mkdir(parents=True, exist_ok=True)

    by_section_path = quality_dir / f"{stem}_eval_by_section.csv"
    results["by_section"].to_csv(by_section_path, index=False)
    logger.info(f"Saved section eval ({len(results['by_section'])} rows) → {by_section_path}")

    by_trial_path = quality_dir / f"{stem}_eval_by_trial.csv"
    results["by_trial"].to_csv(by_trial_path, index=False)
    logger.info(f"Saved trial eval ({len(results['by_trial'])} rows) → {by_trial_path}")

    summary_path = quality_dir / f"{stem}_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results["summary"], f, indent=2)
    logger.info(f"Saved eval summary → {summary_path}")


def save_transition_summary(
    per_trial_df: pd.DataFrame,
    quality_dir: "Path",
    stem: str,
) -> None:
    """Persist per-trial transition summary produced by
    :func:`~bret.features.temporal.summarize_transitions`.

    Writes ``{stem}_transition_summary.csv`` to *quality_dir*.

    Args:
        per_trial_df: DataFrame with columns ``Trial``, ``n_transitions``,
            ``n_jump_induced``, ``n_spontaneous``, ``pct_jump_induced``.
        quality_dir: Output directory (created if needed).
        stem: File stem, e.g. ``'s11r06r'``.
    """
    quality_dir = Path(quality_dir)
    quality_dir.mkdir(parents=True, exist_ok=True)
    out_path = quality_dir / f"{stem}_transition_summary.csv"
    per_trial_df.to_csv(out_path, index=False)
    logger.info(f"Saved transition summary ({len(per_trial_df)} trials) → {out_path}")


# ── keep changepoint writer for ad-hoc use, not called by the pipeline ────────
def _save_percept_changepoints(
    df: pd.DataFrame,
    output_path: Path,
    percept_col: str = "InferredPerceptAggregated",
) -> None:
    """Write collapsed epoch (changepoint) CSV — not used by default pipeline."""
    required = {"Timestamp", "Trial", "Type", "Image1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Changepoint format requires columns {required}; missing: {missing}")
    if percept_col not in df.columns:
        raise ValueError(f"Column '{percept_col}' not found in DataFrame.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    work = (
        df[["Timestamp", "Trial", "Type", "Image1", percept_col]]
        .copy()
        .sort_values(["Trial", "Timestamp"])
        .reset_index(drop=True)
    )
    work["_group"] = (
        (work[percept_col] != work[percept_col].shift())
        | (work["Trial"] != work["Trial"].shift())
    ).cumsum()
    agg = (
        work.groupby(["Trial", "Type", "Image1", percept_col, "_group"], sort=False)
        .agg(onset=("Timestamp", "min"), _last_ts=("Timestamp", "max"))
        .reset_index()
        .sort_values(["Trial", "onset"])
        .reset_index(drop=True)
    )
    agg["_next_onset"] = agg.groupby("Trial")["onset"].shift(-1)
    agg["duration"] = agg["_next_onset"].fillna(agg["_last_ts"]) - agg["onset"]
    grouped = (
        agg.drop(columns=["_group", "_last_ts", "_next_onset"])
        .rename(columns={"Trial": "trial", "Type": "type",
                         "Image1": "leftImage", percept_col: "percept"})
        .sort_values(["trial", "onset"])
        .reset_index(drop=True)
    )
    grouped.to_csv(output_path, index=False)
    logger.info(f"Saved changepoint percept data ({len(grouped)} epochs) → {output_path}")
