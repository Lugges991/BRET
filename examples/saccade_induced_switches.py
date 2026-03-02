"""Investigate how many changes in percept are triggered by changes in fixation points.

Algorithm reference (Kingir et al. 2025):
    A gaze shift was detected at time t if the distance to the new FP decreased
    more than 0.5 deg in the next 100 ms with respect to the previous 100 ms.
    The stability criterion remained the same. Since the saccades of interest are
    due to FP shifts, and are bound to occur soon after the trial onset, we limited
    the time window of saccade detection to 0-1.5 s after trial onsets.

Usage:
    python examples/saccade_induced_switches.py
    python examples/saccade_induced_switches.py --percepts data/sub-12/percepts/s12r04r_percepts.csv
    python examples/saccade_induced_switches.py --all-subjects --data-dir data
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for batch use

# -- resolve project root so script works from any cwd ----------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bret.features.temporal import label_percept_transitions, summarize_transitions
from bret.visualization.percept_plots import plot_percept_timeline, plot_transition_heatmap
from bret.io.writers import save_transition_summary


# -- default parameters (override via CLI) -----------------------------------
DEFAULT_PERCEPTS = project_root / "data/sub-11/percepts/s11r06r_percepts.csv"
DEFAULT_FIGURES_DIR = project_root / "data/figures/transitions"
JUMP_WINDOW_MS = 200       # +/-ms around fixpoint jump to classify switch as jump-induced
SMOOTHING_WINDOW_MS = 400  # median-filter window per trial before detecting transitions
SAMPLING_RATE = 250        # Hz


def analyse_run(
    percepts_path: Path,
    figures_dir: Path,
    verbose: bool = True,
    save_figures: bool = True,
) -> tuple:
    """Run the full transition analysis for a single percepts CSV.

    Args:
        percepts_path: Path to ``{stem}_percepts.csv``.
        figures_dir: Directory to write figures into (only used when save_figures=True).
        verbose: Print per-trial table to stdout.
        save_figures: Whether to generate and save PNG figures.

    Returns:
        Tuple of (per_trial_df, run_summary_dict).
    """
    stem = percepts_path.stem.replace("_percepts", "")
    quality_dir = percepts_path.parent.parent / "quality"

    # -- load ----------------------------------------------------------------
    df = pd.read_csv(percepts_path)

    # -- label transitions ---------------------------------------------------
    df = label_percept_transitions(
        df,
        jump_window_ms=JUMP_WINDOW_MS,
        smoothing_window_ms=SMOOTHING_WINDOW_MS,
        sampling_rate=SAMPLING_RATE,
    )

    # -- summarise -----------------------------------------------------------
    per_trial, run_summary = summarize_transitions(df)
    print(f"  {stem}  ->  {run_summary['n_transitions']} transitions, "
          f"{run_summary['pct_jump_induced']}% jump-induced")

    if verbose:
        print(per_trial.to_string(index=False))

    # -- save CSV summary ----------------------------------------------------
    save_transition_summary(per_trial, quality_dir, stem)

    # -- figures (single-run mode only) --------------------------------------
    if save_figures:
        import matplotlib.pyplot as plt
        figures_dir.mkdir(parents=True, exist_ok=True)
        first_trial = int(df["Trial"].dropna().unique()[0])
        fig_tl = plot_percept_timeline(df, trial=first_trial)
        tl_path = figures_dir / f"{stem}_timeline_trial{first_trial}.png"
        fig_tl.savefig(tl_path, dpi=150, bbox_inches="tight")
        plt.close(fig_tl)
        print(f"  Saved timeline -> {tl_path}")

        fig_hm = plot_transition_heatmap(df)
        hm_path = figures_dir / f"{stem}_transition_heatmap.png"
        fig_hm.savefig(hm_path, dpi=150, bbox_inches="tight")
        plt.close(fig_hm)
        print(f"  Saved heatmap  -> {hm_path}")

    return per_trial, run_summary


def main():
    parser = argparse.ArgumentParser(
        description="Quantify jump-induced vs spontaneous percept transitions."
    )
    parser.add_argument(
        "--percepts", type=Path, default=DEFAULT_PERCEPTS,
        help="Path to a single *_percepts.csv file.",
    )
    parser.add_argument(
        "--all-subjects", action="store_true",
        help="Run across all subjects in --data-dir.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=project_root / "data",
        help="Root data directory (used with --all-subjects).",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR,
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    if args.all_subjects:
        all_files = sorted(args.data_dir.glob("sub-*/percepts/*_percepts.csv"))
        if not all_files:
            print(f"No percepts files found under {args.data_dir}")
            sys.exit(1)
        print(f"Found {len(all_files)} percepts files — running batch (no figures)...\n")
        run_rows = []
        for p in all_files:
            stem = p.stem.replace("_percepts", "")
            try:
                per_trial, run_summary = analyse_run(
                    p, args.figures_dir, verbose=False, save_figures=False
                )
                per_trial["run"] = stem
                run_rows.append({"run": stem, **run_summary})
            except Exception as e:
                print(f"  ERROR {p.name}: {e}")
        if run_rows:
            report = pd.DataFrame(run_rows)
            out = args.data_dir / "transition_report_all_subjects.csv"
            report.to_csv(out, index=False)
            print(f"\nReport ({len(report)} runs) -> {out}")
            print("\n" + report.to_string(index=False))
    else:
        _, _ = analyse_run(args.percepts, args.figures_dir, verbose=True, save_figures=True)


if __name__ == "__main__":
    main()
