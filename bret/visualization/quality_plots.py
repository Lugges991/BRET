"""
Data quality visualization.

Plotting functions for eye tracking data quality assessment.
All functions return matplotlib Figure objects without calling plt.show().
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def plot_preprocessing_quality(
    df: pd.DataFrame,
    trial: Optional[int] = None,
    figsize: tuple = (16, 6),
) -> Figure:
    """
    Plot timeline showing blink/saccade/fixation event coverage for a run.

    Displays coloured bands over time indicating event types, giving a
    visual overview of data quality per trial.

    Args:
        df: Preprocessed DataFrame with Timestamp, Trial, Fixation, Saccade, Blink.
        trial: If given, plot only this trial. Otherwise plots all trials stacked.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    if trial is not None:
        df = df[df["Trial"] == trial].copy()

    trials = sorted(df["Trial"].dropna().unique())
    n_trials = len(trials)

    fig, axes = plt.subplots(n_trials, 1, figsize=(figsize[0], max(2 * n_trials, figsize[1])),
                             sharex=False, squeeze=False)

    colors = {"Blink": "#e74c3c", "Saccade": "#f39c12", "Fixation": "#27ae60"}

    for i, tid in enumerate(trials):
        ax = axes[i, 0]
        tdf = df[df["Trial"] == tid]
        ts = tdf["Timestamp"].values

        for event, color in colors.items():
            if event in tdf.columns:
                mask = tdf[event].values == 1
                ax.fill_between(ts, 0, 1, where=mask, color=color, alpha=0.6, step="mid")

        ax.set_ylabel(f"Trial {int(tid)}", fontsize=9)
        ax.set_yticks([])
        ax.set_xlim(ts[0], ts[-1])

    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle("Event Timeline per Trial", fontsize=13)

    patches = [mpatches.Patch(color=c, label=l, alpha=0.6) for l, c in colors.items()]
    fig.legend(handles=patches, loc="upper right", fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    return fig


def plot_subject_quality_heatmap(
    qc_report: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: tuple = (14, 8),
) -> Figure:
    """
    Heatmap of quality metrics across runs for one or more subjects.

    Args:
        qc_report: DataFrame from compute_subject_quality_report() or
            generate_cross_subject_qc_report(). Must have columns
            'subject', 'run', and the metric columns.
        metrics: Which columns to display. Defaults to a sensible set.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    if metrics is None:
        metrics = [
            "blink_pct", "saccade_pct", "fixation_pct",
            "gaze_std_x", "gaze_std_y", "mean_dist_center",
            "diameter_cv", "pct_within_1deg",
        ]
    metrics = [m for m in metrics if m in qc_report.columns]

    if not metrics:
        logger.warning("No matching metric columns found for heatmap")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    # Build label per row
    qc = qc_report.copy()
    qc["label"] = "s" + qc["subject"].astype(str) + "r" + qc["run"].astype(str).str.zfill(2)
    qc = qc.sort_values(["subject", "run"])

    data = qc[metrics].values.astype(float)

    # Normalise each column to [0, 1] for colour mapping
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    data_norm = (data - col_min) / col_range

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data_norm, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", " ") for m in metrics], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(qc)))
    ax.set_yticklabels(qc["label"].values, fontsize=8)

    # Annotate cells with actual values
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            if np.isnan(val):
                txt = "\u2014"
            elif val > 100:
                txt = f"{val:.0f}"
            else:
                txt = f"{val:.1f}"
            ax.text(c, r, txt, ha="center", va="center", fontsize=7,
                    color="white" if data_norm[r, c] > 0.65 else "black")

    ax.set_title("Data Quality Heatmap (red = potential issue)", fontsize=13)
    fig.colorbar(im, ax=ax, label="Normalised (0=best, 1=worst)", shrink=0.6)
    fig.tight_layout()
    return fig


def plot_blink_rate_comparison(
    qc_report: pd.DataFrame,
    figsize: tuple = (12, 5),
) -> Figure:
    """
    Bar chart comparing blink percentage across subjects and runs.

    Args:
        qc_report: Quality report DataFrame with subject, run, blink_pct.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    qc = qc_report.sort_values(["subject", "run"]).copy()
    qc["label"] = "s" + qc["subject"].astype(str) + "r" + qc["run"].astype(str).str.zfill(2)

    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(qc))
    colors = ["#e74c3c" if b > 15 else "#f39c12" if b > 8 else "#27ae60"
              for b in qc["blink_pct"]]
    ax.bar(x, qc["blink_pct"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(qc["label"], rotation=90, fontsize=7)
    ax.set_ylabel("Blink %")
    ax.set_title("Blink Rate per Run")
    ax.axhline(15, color="#e74c3c", ls="--", lw=0.8, label="High (>15%)")
    ax.axhline(8, color="#f39c12", ls="--", lw=0.8, label="Moderate (>8%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_calibration_quality(
    qc_report: pd.DataFrame,
    figsize: tuple = (12, 5),
) -> Figure:
    """
    Bar chart of % fixation samples within 1° of nearest fixation point.

    Lower values indicate poorer calibration or gaze tracking.

    Args:
        qc_report: Quality report with pct_within_1deg column.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    if "pct_within_1deg" not in qc_report.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "pct_within_1deg not available", ha="center", va="center")
        return fig

    qc = qc_report.sort_values(["subject", "run"]).copy()
    qc["label"] = "s" + qc["subject"].astype(str) + "r" + qc["run"].astype(str).str.zfill(2)

    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(qc))
    vals = qc["pct_within_1deg"].values
    colors = ["#e74c3c" if v < 30 else "#f39c12" if v < 50 else "#27ae60" for v in vals]
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(qc["label"], rotation=90, fontsize=7)
    ax.set_ylabel("% fixation samples within 1°")
    ax.set_title("Calibration / Gaze Tracking Quality")
    ax.axhline(50, color="#f39c12", ls="--", lw=0.8, label="Moderate (<50%)")
    ax.axhline(30, color="#e74c3c", ls="--", lw=0.8, label="Poor (<30%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_quality_vs_accuracy(
    qc_report: pd.DataFrame,
    eval_report: pd.DataFrame,
    quality_metric: str = "blink_pct",
    accuracy_metric: str = "accuracy_2class",
    figsize: tuple = (8, 6),
) -> Figure:
    """
    Scatter plot showing relationship between a quality metric and
    classification accuracy (report runs only).

    Args:
        qc_report: Quality report DataFrame.
        eval_report: Evaluation report (e.g. eval_report_all_subjects.csv)
            with columns: subject, run, accuracy_2class, etc.
        quality_metric: Column from qc_report to use for x-axis.
        accuracy_metric: Column from eval_report to use for y-axis.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    merged = qc_report.merge(
        eval_report[["subject", "run", accuracy_metric]].dropna(),
        on=["subject", "run"],
        how="inner",
    )
    if merged.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No matching runs", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    subjects = sorted(merged["subject"].unique())
    cmap = plt.cm.tab10
    for i, sub in enumerate(subjects):
        sdf = merged[merged["subject"] == sub]
        ax.scatter(
            sdf[quality_metric], sdf[accuracy_metric],
            label=f"sub-{sub}", color=cmap(i % 10), s=60, edgecolors="white", linewidths=0.5,
        )

    ax.set_xlabel(quality_metric.replace("_", " "))
    ax.set_ylabel(accuracy_metric.replace("_", " "))
    ax.set_title(f"{accuracy_metric.replace('_', ' ')} vs {quality_metric.replace('_', ' ')}")
    ax.legend(fontsize=8, ncol=2)

    # Add correlation annotation
    from scipy import stats
    x = merged[quality_metric].values
    y = merged[accuracy_metric].values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() > 3:
        r, p = stats.pearsonr(x[mask], y[mask])
        ax.annotate(f"r = {r:.2f}, p = {p:.3f}", xy=(0.05, 0.95),
                    xycoords="axes fraction", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8))

    fig.tight_layout()
    return fig
