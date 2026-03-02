"""Compare classification configurations and produce performance plots.

Runs 5 reconstruction pipelines on every report run:
  A. Baseline: argmin, no peri-jump exclusion
  B. Argmin + peri-jump exclusion (±50 ms)
  C. Argmin + peri-jump + median smoothing (150 ms)
  D. Log-ratio (τ=0.0) + peri-jump
  E. Log-ratio (τ=0.2) + peri-jump + median smoothing (150 ms)

Section-level accuracy / F1 / MCC compared across configs.
"""

import json
import re
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bret.reconstruction.euclidean import EuclideanReconstructor
from bret.reconstruction.smoothing import TemporalSmoother
from bret.reconstruction.evaluators import ReconstructionEvaluator
from bret.io.loaders import load_percept_reports, join_reported_percepts

logging.basicConfig(level=logging.WARNING)

DATA_DIR = Path("data")
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── configuration matrix ──────────────────────────────────────────────────────

CONFIGS = {
    "A: baseline": dict(
        ratio=False, tau=0.0, peri_jump=False, smooth=False, window=0, delta=0,
    ),
    "B: +peri-jump": dict(
        ratio=False, tau=0.0, peri_jump=True, smooth=False, window=0, delta=0,
    ),
    "C: +smooth 150": dict(
        ratio=False, tau=0.0, peri_jump=True, smooth=True, window=150, delta=0.05,
    ),
    "D: ratio τ=0": dict(
        ratio=True, tau=0.0, peri_jump=True, smooth=False, window=0, delta=0,
    ),
    "E: ratio+smooth": dict(
        ratio=True, tau=0.2, peri_jump=True, smooth=True, window=150, delta=0.05,
    ),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def run_pipeline(df_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply one reconstruction configuration, return df with aggregated col."""
    rec = EuclideanReconstructor(include_mixed=True, ratio_threshold=cfg["tau"])
    smoother = TemporalSmoother(include_mixed=True)

    df = df_raw.copy()

    # sample-level classification
    df = rec.infer_percept_from_closest_fixpoint(df)
    if cfg["ratio"]:
        df = rec.infer_percept_ratio(df)
        percept_col = "InferredPerceptRatio"
    else:
        percept_col = "InferredPercept"

    # section labelling
    df = rec.label_fixpoint_sections(df)

    # peri-jump exclusion
    if cfg["peri_jump"]:
        df = rec.exclude_peri_jump_samples(df, margin_ms=50)

    # temporal smoothing
    if cfg["smooth"]:
        df = smoother.apply_median_filter(
            df, window_size=cfg["window"], percept_col=percept_col,
            threshold=0.5, delta=cfg["delta"],
        )
        percept_col = f"{percept_col}Smoothed"

    # section-level aggregation
    df = rec.aggregate_by_fixation_section(
        df, percept_col=percept_col, exclude_peri_jump=cfg["peri_jump"],
    )
    # Standardize the aggregated column name so the evaluator finds it
    agg_col = f"{percept_col}Aggregated"
    if agg_col != "InferredPerceptAggregated":
        df["InferredPerceptAggregated"] = df[agg_col]
    return df


def evaluate_sections(df: pd.DataFrame, evaluator: ReconstructionEvaluator) -> dict:
    """Evaluate section-level, return flat metrics dict."""
    try:
        res = evaluator.evaluate_by_section(df)
        m2 = res["metrics_2class"]
        return {
            "n_sections": len(res["by_section"]),
            "accuracy": m2["accuracy"],
            "f1_macro": m2["f1_macro"],
            "mcc": m2["mcc"],
        }
    except Exception:
        return {"n_sections": 0, "accuracy": np.nan, "f1_macro": np.nan, "mcc": np.nan}


# ── main loop ─────────────────────────────────────────────────────────────────

print("Comparing reconstruction configurations…\n")

evaluator = ReconstructionEvaluator()
results = []

# iterate over all report-run preprocessed files
for preproc_file in sorted(DATA_DIR.glob("sub-*/preprocessed/*r_preprocessed.csv")):
    # only report runs (filename ends ...r_preprocessed.csv, not nr)
    if "nr_preprocessed" in preproc_file.name:
        continue

    subject_dir = preproc_file.parent.parent
    sub_match = re.search(r"sub-(\d+)", str(subject_dir))
    run_match = re.search(r"r(\d+)r_preprocessed", preproc_file.name)
    if not (sub_match and run_match):
        continue

    sub = int(sub_match.group(1))
    run = int(run_match.group(1))

    # Load preprocessed data
    df_raw = pd.read_csv(preproc_file)
    if len(df_raw) == 0:
        continue

    # Join ground-truth percepts
    try:
        percept_data = load_percept_reports(subject_dir, run)
        df_raw = join_reported_percepts(df_raw, percept_data)
    except FileNotFoundError:
        continue

    if "ReportedPercept" not in df_raw.columns or df_raw["ReportedPercept"].isna().all():
        continue

    print(f"  sub-{sub} run-{run:02d} ({len(df_raw)} samples) …", end="")

    for cfg_name, cfg in CONFIGS.items():
        df_out = run_pipeline(df_raw, cfg)
        metrics = evaluate_sections(df_out, evaluator)
        results.append({
            "subject": sub,
            "run": run,
            "config": cfg_name,
            **metrics,
        })

    print(" done")

if not results:
    print("No report runs found.")
    raise SystemExit(1)

df_results = pd.DataFrame(results)
df_valid = df_results.dropna(subset=["accuracy"])

# Save raw comparison table
out_csv = DATA_DIR / "config_comparison.csv"
df_valid.to_csv(out_csv, index=False)
print(f"\nComparison table saved → {out_csv}")

# ── summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 85)
print("CONFIG COMPARISON — section-level 2-class metrics (mean ± std across runs)")
print("=" * 85)
summary = (
    df_valid.groupby("config")
    .agg(
        n_runs=("accuracy", "count"),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        f1_mean=("f1_macro", "mean"),
        f1_std=("f1_macro", "std"),
        mcc_mean=("mcc", "mean"),
        mcc_std=("mcc", "std"),
    )
    .reset_index()
)
for _, row in summary.iterrows():
    print(
        f"  {row['config']:22s}  "
        f"Acc={row['acc_mean']:.3f}±{row['acc_std']:.3f}  "
        f"F1={row['f1_mean']:.3f}±{row['f1_std']:.3f}  "
        f"MCC={row['mcc_mean']:.3f}±{row['mcc_std']:.3f}  "
        f"(N={int(row['n_runs'])})"
    )

# ── Figure 1: grouped bar chart of mean metrics by config ─────────────────────
config_names = summary["config"].tolist()
x = np.arange(len(config_names))
width = 0.25

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.bar(x - width, summary["acc_mean"], width, yerr=summary["acc_std"],
        label="Accuracy", color="#4C72B0", alpha=0.85, capsize=3)
ax1.bar(x,         summary["f1_mean"],  width, yerr=summary["f1_std"],
        label="F1 macro", color="#55A868", alpha=0.85, capsize=3)
ax1.bar(x + width, summary["mcc_mean"], width, yerr=summary["mcc_std"],
        label="MCC", color="#C44E52", alpha=0.85, capsize=3)

ax1.set_xticks(x)
ax1.set_xticklabels(config_names, rotation=20, ha="right", fontsize=9)
ax1.set_ylabel("Score")
ax1.set_ylim(0, 1.15)
ax1.set_title("Reconstruction quality by configuration (section-level, 2-class)")
ax1.legend(loc="upper left")
ax1.axhline(0.5, color="gray", ls="--", lw=0.8)
ax1.grid(axis="y", lw=0.4, alpha=0.5)
fig1.tight_layout()
fig1_path = FIGURES_DIR / "config_comparison_bars.png"
fig1.savefig(fig1_path, dpi=150)
plt.close(fig1)
print(f"\nFigure 1 saved → {fig1_path}")

# ── Figure 2: per-subject accuracy comparison (baseline vs best) ──────────────
pivot_acc = df_valid.pivot_table(
    index="subject", columns="config", values="accuracy", aggfunc="mean",
).sort_index()

fig2, ax2 = plt.subplots(figsize=(10, 5))
subs = pivot_acc.index.values
x = np.arange(len(subs))
n_cfg = len(config_names)
total_width = 0.75
bw = total_width / n_cfg

colors = ["#AAAAAA", "#4C72B0", "#55A868", "#DD8452", "#C44E52"]
for i, cfg in enumerate(config_names):
    if cfg in pivot_acc.columns:
        vals = pivot_acc[cfg].values
        ax2.bar(x + i * bw - total_width / 2 + bw / 2, vals, bw,
                label=cfg, color=colors[i % len(colors)], alpha=0.85)

ax2.set_xticks(x)
ax2.set_xticklabels([f"sub-{s}" for s in subs], rotation=30, ha="right")
ax2.set_ylabel("Mean accuracy (2-class)")
ax2.set_ylim(0, 1.1)
ax2.set_title("Per-subject accuracy across configurations")
ax2.legend(fontsize=7, ncol=2)
ax2.axhline(0.5, color="gray", ls="--", lw=0.8)
ax2.grid(axis="y", lw=0.4, alpha=0.5)
fig2.tight_layout()
fig2_path = FIGURES_DIR / "config_comparison_per_subject.png"
fig2.savefig(fig2_path, dpi=150)
plt.close(fig2)
print(f"Figure 2 saved → {fig2_path}")

# ── Figure 3: improvement Δ relative to baseline ─────────────────────────────
baseline_col = config_names[0]  # "A: baseline"
fig3, ax3 = plt.subplots(figsize=(10, 5))

for i, cfg in enumerate(config_names[1:], 1):
    if cfg in pivot_acc.columns and baseline_col in pivot_acc.columns:
        delta = pivot_acc[cfg].values - pivot_acc[baseline_col].values
        ax3.bar(x + (i - 1) * bw - total_width / 2 + bw / 2, delta, bw,
                label=cfg, color=colors[i % len(colors)], alpha=0.85)

ax3.set_xticks(x)
ax3.set_xticklabels([f"sub-{s}" for s in subs], rotation=30, ha="right")
ax3.set_ylabel("Δ accuracy vs baseline")
ax3.axhline(0, color="black", lw=0.8)
ax3.set_title("Accuracy change relative to baseline (A: argmin only)")
ax3.legend(fontsize=7, ncol=2)
ax3.grid(axis="y", lw=0.4, alpha=0.5)
fig3.tight_layout()
fig3_path = FIGURES_DIR / "config_comparison_delta.png"
fig3.savefig(fig3_path, dpi=150)
plt.close(fig3)
print(f"Figure 3 saved → {fig3_path}")

# ── Figure 4: box plots by config ────────────────────────────────────────────
fig4, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, metric, label in zip(
    axes, ["accuracy", "f1_macro", "mcc"], ["Accuracy", "F1 macro", "MCC"]
):
    data_by_cfg = [
        df_valid.loc[df_valid["config"] == cfg, metric].dropna().values
        for cfg in config_names
    ]
    bp = ax.boxplot(data_by_cfg, tick_labels=config_names, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_title(label)
    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

fig4.suptitle("Metric distributions across runs by configuration", y=1.01)
fig4.tight_layout()
fig4_path = FIGURES_DIR / "config_comparison_boxplots.png"
fig4.savefig(fig4_path, dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"Figure 4 saved → {fig4_path}")
