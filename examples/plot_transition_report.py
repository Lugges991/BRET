"""Presentation figure from transition_report_all_subjects.csv.

Two-panel figure:
  Left  — per-subject % jump-induced transitions (report vs no-report, box plot)
  Right — group-level stacked bar: mean jump-induced vs spontaneous per subject

Usage:
    python examples/plot_transition_report.py
    python examples/plot_transition_report.py --input data/transition_report_all_subjects.csv
    python examples/plot_transition_report.py --output data/figures/transitions_summary.png
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_INPUT  = project_root / "data/transition_report_all_subjects.csv"
DEFAULT_OUTPUT = project_root / "data/figures/transitions_summary.png"

# ── colours ───────────────────────────────────────────────────────────────────
C_JUMP  = "#D62728"   # red  — jump-induced
C_SPON  = "#2CA02C"   # green — spontaneous
C_REP   = "#4C72B0"   # blue  — report runs
C_NR    = "#DD8452"   # orange — no-report runs


def parse_run(run: str):
    """Return (subject_id, run_type) from e.g. 's11r05nr' or 's11r06r'."""
    m = re.match(r"s(\d+)r\d+(.*)$", run)
    if not m:
        return None, None
    subject = int(m.group(1))
    suffix  = m.group(2)
    run_type = "no-report" if suffix.startswith("nr") else "report"
    return subject, run_type


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    parsed = df["run"].apply(parse_run)
    df["subject"]  = parsed.apply(lambda x: x[0])
    df["run_type"] = parsed.apply(lambda x: x[1])
    df = df.dropna(subset=["subject"])
    df["subject"] = df["subject"].astype(int)
    # transitions per trial (normalise for different n_trials)
    df["transitions_per_trial"] = df["n_transitions"] / df["n_trials"]
    return df


def make_figure(df: pd.DataFrame, out_path: Path):
    subjects = sorted(df["subject"].unique())
    x = np.arange(len(subjects))

    # per-subject mean % jump-induced (for stacked bar)
    sub_means = (
        df.groupby("subject")
        .agg(
            mean_pct_jump=("pct_jump_induced", "mean"),
            mean_pct_spon=("pct_jump_induced", lambda s: 100 - s.mean()),
        )
        .loc[subjects]
    )

    # split by run type for box plot
    rep_by_sub = [
        df.loc[(df["subject"] == s) & (df["run_type"] == "report"), "pct_jump_induced"].values
        for s in subjects
    ]
    nr_by_sub  = [
        df.loc[(df["subject"] == s) & (df["run_type"] == "no-report"), "pct_jump_induced"].values
        for s in subjects
    ]

    # overall medians for annotation
    overall_report   = df.loc[df["run_type"] == "report",    "pct_jump_induced"]
    overall_noreport = df.loc[df["run_type"] == "no-report", "pct_jump_induced"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.patch.set_facecolor("white")

    # ── Left panel: grouped box plot per subject ───────────────────────────────
    bw = 0.35   # box width
    gap = 0.08

    for i, s in enumerate(subjects):
        r_data = rep_by_sub[i]
        n_data = nr_by_sub[i]

        def _box(ax, data, xpos, color):
            if len(data) == 0:
                return
            bp = ax.boxplot(
                data,
                positions=[xpos],
                widths=bw,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker="o", markersize=3, alpha=0.5,
                                markerfacecolor=color, markeredgecolor=color),
                boxprops=dict(facecolor=color, alpha=0.65, linewidth=1.2),
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.2),
            )

        _box(ax1, r_data,  i - bw / 2 - gap / 2, C_REP)
        _box(ax1, n_data,  i + bw / 2 + gap / 2, C_NR)

    # overall median lines
    ax1.axhline(overall_report.median(),   color=C_REP, linestyle="--",
                linewidth=1.2, alpha=0.7,
                label=f"Report median ({overall_report.median():.1f}%)")
    ax1.axhline(overall_noreport.median(), color=C_NR,  linestyle="--",
                linewidth=1.2, alpha=0.7,
                label=f"No-report median ({overall_noreport.median():.1f}%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"sub-{s}" for s in subjects], rotation=40, ha="right", fontsize=9)
    ax1.set_ylabel("Jump-induced transitions (%)", fontsize=12)
    ax1.set_title("Jump-induced transitions per run\n(report vs no-report)", fontsize=13, pad=10)
    ax1.legend(
        handles=[
            mpatches.Patch(color=C_REP, alpha=0.75, label="Report"),
            mpatches.Patch(color=C_NR,  alpha=0.75, label="No-report"),
        ],
        loc="upper right", fontsize=10,
    )
    ax1.set_ylim(0, max(df["pct_jump_induced"].max() * 1.12, 65))
    ax1.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Right panel: stacked bar — mean % jump vs spontaneous ─────────────────
    bars_spon = ax2.bar(
        x, sub_means["mean_pct_spon"],
        color=C_SPON, alpha=0.8, label="Spontaneous",
    )
    bars_jump = ax2.bar(
        x, sub_means["mean_pct_jump"],
        bottom=sub_means["mean_pct_spon"],
        color=C_JUMP, alpha=0.8, label="Jump-induced",
    )

    # annotate jump % on top of each bar
    for xi, (_, row) in zip(x, sub_means.iterrows()):
        ax2.text(xi, 101, f"{row['mean_pct_jump']:.0f}%",
                 ha="center", va="bottom", fontsize=7.5, color=C_JUMP)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"sub-{s}" for s in subjects], rotation=40, ha="right", fontsize=9)
    ax2.set_ylabel("Mean proportion of transitions (%)", fontsize=11)
    ax2.set_title("Avg. transition breakdown\nper subject", fontsize=13, pad=10)
    ax2.set_ylim(0, 115)
    ax2.axhline(
        100 - df["pct_jump_induced"].mean(), color=C_JUMP,
        linestyle="--", linewidth=1.2, alpha=0.7,
        label=f"Group mean ({df['pct_jump_induced'].mean():.1f}%)",
    )
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── overall stats subtitle ────────────────────────────────────────────────
    n_runs = len(df)
    n_subs = df["subject"].nunique()
    total_trans = df["n_transitions"].sum()
    total_jump  = df["n_jump_induced"].sum()
    fig.suptitle(
        f"Fixation-jump-induced vs spontaneous percept transitions  "
        f"({n_subs} subjects · {n_runs} runs · {total_trans:,} transitions total · "
        f"{100*total_jump/total_trans:.1f}% jump-induced)",
        fontsize=12, y=1.01,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = load(args.input)
    print(f"Loaded {len(df)} runs, {df['subject'].nunique()} subjects")
    make_figure(df, args.output)


if __name__ == "__main__":
    main()
