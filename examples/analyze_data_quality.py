#!/usr/bin/env python3
"""
Data Quality Analysis across all subjects.

Generates a cross-subject quality report (CSV) and quality visualizations
(heatmap, blink rate, calibration quality, quality-vs-accuracy scatter).

Usage:
    python examples/analyze_data_quality.py
    python examples/analyze_data_quality.py --subjects 14,15,18
    python examples/analyze_data_quality.py --output-dir data/figures/quality
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bret.quality.reports import generate_cross_subject_qc_report
from bret.visualization.quality_plots import (
    plot_subject_quality_heatmap,
    plot_blink_rate_comparison,
    plot_calibration_quality,
    plot_quality_vs_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Analyze eye tracking data quality")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--subjects", type=str, default=None,
        help="Comma-separated subject numbers (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to save figures (default: data/figures/quality)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir / "figures" / "quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = None
    if args.subjects:
        subjects = [int(s) for s in args.subjects.split(",")]

    # ── 1. Generate cross-subject quality report ──────────────────────
    logger.info("Computing quality metrics across all subjects...")
    qc = generate_cross_subject_qc_report(
        data_dir, subjects=subjects,
        output_path=data_dir / "qc_report_all_subjects.csv",
    )

    if qc.empty:
        logger.error("No quality data generated — check that preprocessed/ dirs exist")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("QUALITY REPORT SUMMARY")
    print("=" * 70)
    print(f"Subjects: {sorted(qc['subject'].unique())}")
    print(f"Total runs: {len(qc)}")
    print()

    # Per-subject averages
    summary = qc.groupby("subject").agg({
        "blink_pct": "mean",
        "saccade_pct": "mean",
        "mean_dist_center": "mean",
        "pct_within_1deg": "mean",
    }).round(1)
    summary.columns = ["Avg blink %", "Avg saccade %", "Avg dist center (°)", "Avg % within 1°"]
    print(summary.to_string())
    print()

    # Flag problematic runs
    bad_blink = qc[qc["blink_pct"] > 15]
    bad_cal = qc[qc["pct_within_1deg"] < 30] if "pct_within_1deg" in qc.columns else pd.DataFrame()

    if not bad_blink.empty:
        print(f"⚠  Runs with >15% blinks ({len(bad_blink)}):")
        for _, row in bad_blink.iterrows():
            print(f"   sub-{int(row['subject'])} run{int(row['run']):02d}: {row['blink_pct']:.1f}%")
        print()

    if not bad_cal.empty:
        print(f"⚠  Runs with <30% samples within 1° ({len(bad_cal)}):")
        for _, row in bad_cal.iterrows():
            print(f"   sub-{int(row['subject'])} run{int(row['run']):02d}: {row['pct_within_1deg']:.1f}%")
        print()

    # ── 2. Generate visualizations ────────────────────────────────────
    logger.info("Generating quality plots...")

    # Heatmap
    fig = plot_subject_quality_heatmap(qc)
    fig.savefig(output_dir / "quality_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved quality_heatmap.png")

    # Blink rate bars
    fig = plot_blink_rate_comparison(qc)
    fig.savefig(output_dir / "blink_rate_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved blink_rate_comparison.png")

    # Calibration quality
    if "pct_within_1deg" in qc.columns:
        fig = plot_calibration_quality(qc)
        fig.savefig(output_dir / "calibration_quality.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved calibration_quality.png")

    # Quality vs accuracy scatter (if eval data exists)
    eval_path = data_dir / "eval_report_all_subjects.csv"
    if eval_path.exists():
        eval_df = pd.read_csv(eval_path)
        for qmetric in ["blink_pct", "pct_within_1deg", "mean_dist_center"]:
            if qmetric not in qc.columns:
                continue
            fig = plot_quality_vs_accuracy(qc, eval_df, quality_metric=qmetric)
            fname = f"quality_vs_accuracy_{qmetric}.png"
            fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved {fname}")

    print(f"\nAll figures saved to {output_dir}/")
    print(f"Quality report saved to {data_dir / 'qc_report_all_subjects.csv'}")


if __name__ == "__main__":
    main()
