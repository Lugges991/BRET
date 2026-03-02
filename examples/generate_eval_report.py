"""Aggregate evaluation summaries across all subjects and print a report."""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("data")

rows = []
for summary_path in sorted(DATA_DIR.glob("*/quality/*_eval_summary.json")):
    sub_match = re.search(r"sub-(\d+)", str(summary_path))
    run_match = re.search(r"s\d+r(\d+)r_eval", summary_path.name)
    if not (sub_match and run_match):
        continue
    sub = int(sub_match.group(1))
    run = int(run_match.group(1))
    with open(summary_path) as f:
        s = json.load(f)
    rows.append({"subject": sub, "run": run, **s})

df = pd.DataFrame(rows).sort_values(["subject", "run"])

# ── per-run table ──────────────────────────────────────────────────────────────
cols = ["subject", "run", "n_sections",
        "accuracy_2class", "f1_macro_2class", "mcc_2class",
        "accuracy_3class", "f1_macro_3class", "mcc_3class"]
run_table = df[cols].copy()
for c in ["accuracy_2class", "f1_macro_2class", "mcc_2class",
          "accuracy_3class", "f1_macro_3class", "mcc_3class"]:
    run_table[c] = run_table[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

print("=" * 90)
print("BRET — Reconstruction Evaluation Report (section-level, 2-class = house/face only)")
print("=" * 90)
print()
print(run_table.to_string(index=False))

# ── per-subject summary ────────────────────────────────────────────────────────
sub_summary = (
    df.groupby("subject")
    .agg(
        n_runs=("run", "count"),
        mean_acc_2class=("accuracy_2class", "mean"),
        mean_f1_2class=("f1_macro_2class", "mean"),
        mean_mcc_2class=("mcc_2class", "mean"),
    )
    .reset_index()
)

print()
print("─" * 60)
print("Per-subject averages (2-class, excluding runs with empty preprocessed files)")
print("─" * 60)
print(sub_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ── overall summary ────────────────────────────────────────────────────────────
valid = df.dropna(subset=["accuracy_2class"])
print()
print("─" * 60)
print(f"Overall (N={len(valid)} runs, {valid['subject'].nunique()} subjects)")
print("─" * 60)
for label, metric in [("Accuracy", "accuracy_2class"),
                       ("F1 macro", "f1_macro_2class"),
                       ("MCC",      "mcc_2class")]:
    vals = valid[metric].dropna()
    print(f"  {label:12s}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
          f"min={vals.min():.3f}  max={vals.max():.3f}")

print()
print("Note: NaN rows = empty preprocessed files (sub-15 r09/r11, sub-19 r03).")
print("      sub-17 excluded (preprocessing pending).")

# ── save to CSV ───────────────────────────────────────────────────────────────
out_path = DATA_DIR / "eval_report_all_subjects.csv"
df[cols].to_csv(out_path, index=False)
print(f"\nFull results saved → {out_path}")

# ── figures ───────────────────────────────────────────────────────────────────
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

COLORS = {
    "house": "#4C72B0",
    "face": "#DD8452",
    "2class": "#4C72B0",
    "3class": "#C44E52",
    "mcc": "#55A868",
}

valid = df.dropna(subset=["accuracy_2class"]).copy()
valid["subject_label"] = "sub-" + valid["subject"].astype(str)

# ── Figure 1: per-subject mean 2-class vs 3-class accuracy ───────────────────
sub_acc = (
    valid.groupby("subject_label")
    .agg(
        acc_2=("accuracy_2class", "mean"),
        acc_3=("accuracy_3class", "mean"),
    )
    .sort_index()
)

x = np.arange(len(sub_acc))
width = 0.38

fig1, ax1 = plt.subplots(figsize=(10, 4.5))
bars2 = ax1.bar(x - width / 2, sub_acc["acc_2"], width,
                label="2-class (house/face)", color=COLORS["2class"], alpha=0.85)
bars3 = ax1.bar(x + width / 2, sub_acc["acc_3"], width,
                label="3-class (+mixed)", color=COLORS["3class"], alpha=0.85)
ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance (0.5)")

ax1.set_xticks(x)
ax1.set_xticklabels(sub_acc.index, rotation=30, ha="right")
ax1.set_ylabel("Mean section-level accuracy")
ax1.set_ylim(0, 1.05)
ax1.set_title("Per-subject mean reconstruction accuracy (2-class vs 3-class)")
ax1.legend(framealpha=0.9)
ax1.grid(axis="y", linewidth=0.5, alpha=0.6)

# annotate bar tops
for bar in list(bars2) + list(bars3):
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
             f"{h:.2f}", ha="center", va="bottom", fontsize=7)

fig1.tight_layout()
fig1_path = FIGURES_DIR / "accuracy_per_subject.png"
fig1.savefig(fig1_path, dpi=150)
print(f"Figure 1 saved → {fig1_path}")
plt.close(fig1)

# ── Figure 2: metric distributions across runs (box plot) ────────────────────
metrics = {
    "accuracy_2class": "Accuracy (2-class)",
    "f1_macro_2class": "F1 macro (2-class)",
    "mcc_2class": "MCC (2-class)",
}

fig2, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
for ax, (col, label) in zip(axes, metrics.items()):
    data_by_sub = [
        grp[col].dropna().values
        for _, grp in valid.groupby("subject_label")
    ]
    labels = [name for name, _ in valid.groupby("subject_label")]
    bp = ax.boxplot(data_by_sub, tick_labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["2class"])
        patch.set_alpha(0.6)
    ax.set_title(label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linewidth=0.5, alpha=0.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

fig2.suptitle("Run-level metric distributions per subject (2-class)", y=1.01)
fig2.tight_layout()
fig2_path = FIGURES_DIR / "metric_distributions.png"
fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
print(f"Figure 2 saved → {fig2_path}")
plt.close(fig2)

# ── Figure 3: accuracy heatmap (subjects × runs) ─────────────────────────────
pivot = valid.pivot_table(
    index="subject_label", columns="run", values="accuracy_2class"
).sort_index()

fig3, ax3 = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.9), max(4, len(pivot) * 0.6)))
im = ax3.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)

ax3.set_xticks(range(len(pivot.columns)))
ax3.set_xticklabels([f"r{c:02d}" for c in pivot.columns])
ax3.set_yticks(range(len(pivot.index)))
ax3.set_yticklabels(pivot.index)
ax3.set_xlabel("Run")
ax3.set_ylabel("Subject")
ax3.set_title("2-class reconstruction accuracy per run (section-level)")

# annotate cells
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            color = "black" if 0.45 < val < 0.85 else "white"
            ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=8, color=color)

fig3.colorbar(im, ax=ax3, label="Accuracy")
fig3.tight_layout()
fig3_path = FIGURES_DIR / "accuracy_heatmap.png"
fig3.savefig(fig3_path, dpi=150)
print(f"Figure 3 saved → {fig3_path}")
plt.close(fig3)
