"""
Quality control report generation.

Produces per-subject and cross-subject quality reports as CSV/JSON,
combining metrics from checks.py.
"""

import pandas as pd
from pathlib import Path
import json
import logging
from typing import List, Optional

from bret.quality.checks import compute_subject_quality_report

logger = logging.getLogger(__name__)


def generate_qc_report(
    subject_dir: Path,
    output_path: Path = None,
    format: str = "csv",
) -> pd.DataFrame:
    """
    Generate quality control report for a single subject.

    Computes all quality metrics for every preprocessed run and saves
    the report to disk.

    Args:
        subject_dir: Path to subject directory (e.g. ``data/sub-11/``).
        output_path: Where to save the report. Defaults to
            ``subject_dir/quality/qc_report.csv`` (or ``.json``).
        format: ``'csv'`` or ``'json'``.

    Returns:
        DataFrame with per-run quality metrics.
    """
    subject_dir = Path(subject_dir)
    report = compute_subject_quality_report(subject_dir)

    if report.empty:
        logger.warning(f"Empty QC report for {subject_dir.name}")
        return report

    if output_path is None:
        quality_dir = subject_dir / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        ext = "json" if format == "json" else "csv"
        output_path = quality_dir / f"qc_report.{ext}"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        report.to_json(output_path, orient="records", indent=2)
    else:
        report.to_csv(output_path, index=False)

    logger.info(f"Saved QC report ({len(report)} runs) to {output_path}")
    return report


def generate_cross_subject_qc_report(
    data_dir: Path,
    subjects: Optional[List[int]] = None,
    output_path: Path = None,
) -> pd.DataFrame:
    """
    Generate a combined quality report across all subjects.

    Args:
        data_dir: Root data directory containing sub-{N}/ folders.
        subjects: List of subject numbers to include. If None, auto-discovers.
        output_path: Where to save the combined report. Defaults to
            ``data_dir/qc_report_all_subjects.csv``.

    Returns:
        Combined DataFrame with per-run quality metrics across subjects.
    """
    data_dir = Path(data_dir)

    if subjects is None:
        sub_dirs = sorted(data_dir.glob("sub-*"))
    else:
        sub_dirs = [data_dir / f"sub-{s}" for s in subjects]

    all_reports = []
    for sd in sub_dirs:
        if not sd.is_dir():
            continue
        report = compute_subject_quality_report(sd)
        if not report.empty:
            all_reports.append(report)

    if not all_reports:
        logger.warning("No quality data found")
        return pd.DataFrame()

    combined = pd.concat(all_reports, ignore_index=True)

    if output_path is None:
        output_path = data_dir / "qc_report_all_subjects.csv"

    output_path = Path(output_path)
    combined.to_csv(output_path, index=False)
    logger.info(
        f"Saved cross-subject QC report ({len(combined)} runs, "
        f"{combined['subject'].nunique()} subjects) to {output_path}"
    )
    return combined
