"""
Main command-line interface for BRET package.
"""

import argparse
import sys
from pathlib import Path
import logging
import re

from bret.utils.config_loader import load_config
from bret.utils.logging_setup import setup_logging
from bret.preprocessing.pipeline import PreprocessingPipeline
from bret.reconstruction.euclidean import EuclideanReconstructor
from bret.reconstruction.smoothing import TemporalSmoother
from bret.reconstruction.evaluators import ReconstructionEvaluator
from bret.io.writers import save_percept_data, save_evaluation_results
from bret.io.loaders import load_percept_reports, join_reported_percepts

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BRET: Binocular Rivalry Eye Tracking Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── preprocess ────────────────────────────────────────────────────────────
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess raw .asc files for one subject",
    )
    preprocess_parser.add_argument(
        "--subject-dir", dest="subject_dir", type=str, required=True,
        help="Path to subject directory containing .asc files",
    )
    preprocess_parser.add_argument("--config", type=str, help="Custom config file")

    # ── reconstruct ───────────────────────────────────────────────────────────
    reconstruct_parser = subparsers.add_parser(
        "reconstruct",
        help="Classify percepts from preprocessed gaze data for one subject",
    )
    reconstruct_parser.add_argument(
        "--subject-dir", dest="subject_dir", type=str, required=True,
        help="Path to subject directory (must contain a preprocessed/ sub-folder)",
    )
    reconstruct_parser.add_argument(
        "--include-mixed", dest="include_mixed", action="store_true", default=True,
        help="Also compute 3-class InferredPerceptMixed column (default: on)",
    )
    reconstruct_parser.add_argument(
        "--aggregate", action="store_true", default=True,
        help="Add FixpointSection + InferredPerceptAggregated columns (default: on)",
    )
    reconstruct_parser.add_argument(
        "--evaluate", action="store_true", default=False,
        help="Evaluate reconstruction quality on report runs (saves to quality/)",
    )
    reconstruct_parser.add_argument("--config", type=str, help="Custom config file")

    # ── process (full pipeline) ───────────────────────────────────────────────
    process_parser = subparsers.add_parser(
        "process",
        help="Run full pipeline (preprocess + reconstruct) for one subject",
    )
    process_parser.add_argument(
        "--subject-dir", dest="subject_dir", type=str, required=True,
    )
    process_parser.add_argument("--config", type=str, help="Custom config file")

    # ── validate ──────────────────────────────────────────────────────────────
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate data quality",
    )
    validate_parser.add_argument("--subject", type=int, required=True)
    validate_parser.add_argument("--run", type=int, help="Specific run (optional)")

    # ── batch ─────────────────────────────────────────────────────────────────
    batch_parser = subparsers.add_parser(
        "batch",
        help="Run preprocess + reconstruct for multiple subjects",
    )
    batch_parser.add_argument(
        "--data-dir", dest="data_dir", type=str, default="data",
        help="Root data directory (default: data/)",
    )
    batch_parser.add_argument(
        "--subjects", type=str,
        help="Comma-separated subject numbers (e.g. 11,12,13). "
             "Omit to process all sub-* directories.",
    )
    batch_parser.add_argument(
        "--preprocess-only", dest="preprocess_only", action="store_true",
        help="Only run preprocessing, skip reconstruction",
    )
    batch_parser.add_argument(
        "--reconstruct-only", dest="reconstruct_only", action="store_true",
        help="Only run reconstruction (preprocessed files must already exist)",
    )
    batch_parser.add_argument(
        "--evaluate", action="store_true", default=False,
        help="Evaluate reconstruction quality on report runs (saves to quality/)",
    )
    batch_parser.add_argument(
        "--events", action="store_true", default=False,
        help="Generate events.tsv files for all runs after reconstruction",
    )
    batch_parser.add_argument(
        "--exclude-jump-induced", dest="exclude_jump_induced", action="store_true",
        default=False,
        help="Exclude jump-induced percept epochs from events.tsv (merges into preceding)",
    )
    batch_parser.add_argument(
        "--n-dummies", dest="n_dummies", type=int, default=5,
        help="Number of dummy TRs discarded before each run (default: 5)",
    )
    batch_parser.add_argument(
        "--tr", dest="tr", type=float, default=1.75,
        help="Repetition time in seconds (default: 1.75)",
    )
    batch_parser.add_argument(
        "--trial-dur", dest="trial_dur", type=float, default=120.0,
        help="Expected trial duration in seconds (default: 120.0)",
    )
    batch_parser.add_argument(
        "--iti", dest="iti", type=float, default=20.0,
        help="Inter-trial interval in seconds (default: 20.0)",
    )
    batch_parser.add_argument(
        "--min-epoch-duration", dest="min_epoch_duration", type=float, default=None,
        help="Minimum epoch duration (s); shorter epochs are merged into neighbours. "
             "Defaults to TR.",
    )
    batch_parser.add_argument("--config", type=str, help="Custom config file")

    # ── events ────────────────────────────────────────────────────────────────
    events_parser = subparsers.add_parser(
        "events",
        help="Generate events.tsv files for all runs in a subject directory",
    )
    events_parser.add_argument(
        "--subject-dir", dest="subject_dir", type=str, required=True,
        help="Path to subject directory (must contain a percepts/ sub-folder)",
    )
    events_parser.add_argument(
        "--exclude-jump-induced", dest="exclude_jump_induced", action="store_true",
        default=False,
        help="Exclude jump-induced percept epochs (merges duration into preceding epoch)",
    )
    events_parser.add_argument(
        "--n-dummies", dest="n_dummies", type=int, default=5,
        help="Number of dummy TRs discarded before the run (default: 5)",
    )
    events_parser.add_argument(
        "--tr", dest="tr", type=float, default=1.75,
        help="Repetition time in seconds (default: 1.75)",
    )
    events_parser.add_argument(
        "--trial-dur", dest="trial_dur", type=float, default=120.0,
        help="Expected trial duration in seconds (default: 120.0)",
    )
    events_parser.add_argument(
        "--iti", dest="iti", type=float, default=20.0,
        help="Inter-trial interval in seconds (default: 20.0)",
    )
    events_parser.add_argument(
        "--min-epoch-duration", dest="min_epoch_duration", type=float, default=None,
        help="Minimum epoch duration (s); shorter epochs are merged into neighbours. "
             "Defaults to TR.",
    )
    events_parser.add_argument("--config", type=str, help="Custom config file")

    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (use -vv for DEBUG)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Logging level
    if args.verbose >= 2:
        log_level = "DEBUG"
    elif args.verbose == 1:
        log_level = "INFO"
    else:
        log_level = "WARNING"
    setup_logging(level=log_level)

    config_path = getattr(args, "config", None)
    config = load_config(Path(config_path) if config_path else None)

    if args.command == "preprocess":
        run_preprocess(args, config)
    elif args.command == "reconstruct":
        run_reconstruct(args, config)
    elif args.command == "process":
        run_process(args, config)
    elif args.command == "validate":
        run_validate(args, config)
    elif args.command == "batch":
        run_batch(args, config)
    elif args.command == "events":
        run_events(args, config)
    else:
        parser.print_help()
        sys.exit(1)


# ── command implementations ────────────────────────────────────────────────────

def run_preprocess(args, config=None):
    """Preprocess all .asc files in a subject directory."""
    if config is None:
        config = load_config()

    subject_dir = Path(args.subject_dir)
    logger.info(f"Preprocessing {subject_dir}")
    output_dir = subject_dir / "preprocessed"
    output_dir.mkdir(exist_ok=True)

    pipeline = PreprocessingPipeline(config)
    pipeline.process_subject(subject_dir=subject_dir, output_dir=output_dir)
    print(f"Preprocessing complete → {output_dir}")


def run_reconstruct(args, config=None):
    """Classify percepts for all preprocessed files in a subject directory."""
    if config is None:
        config = load_config()

    subject_dir = Path(args.subject_dir)
    preprocessed_dir = subject_dir / "preprocessed"
    if not preprocessed_dir.exists():
        print(
            f"ERROR: {preprocessed_dir} does not exist. "
            "Run 'bret preprocess' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    files = sorted(preprocessed_dir.glob("*_preprocessed.csv"))
    if not files:
        print(f"No *_preprocessed.csv files found in {preprocessed_dir}", file=sys.stderr)
        sys.exit(1)

    percepts_dir = subject_dir / "percepts"
    percepts_dir.mkdir(exist_ok=True)

    include_mixed = getattr(args, "include_mixed", True)
    aggregate = getattr(args, "aggregate", True)
    do_evaluate = getattr(args, "evaluate", False)

    # Read reconstruction parameters from config
    use_ratio = config.get("use_ratio_classifier", False)
    ratio_threshold = config.get("ratio_threshold", 0.0)
    peri_jump_margin = config.get("peri_jump_margin_ms", 50)
    exclude_peri_jump = config.get("exclude_peri_jump", True)
    do_smooth = config.get("apply_smoothing", False)
    smooth_method = config.get("smoothing_method", "median")
    smooth_window = config.get("smoothing_window_ms", 150)
    smooth_delta = config.get("smoothing_delta", 0.05)

    rec = EuclideanReconstructor(
        include_mixed=include_mixed,
        ratio_threshold=ratio_threshold,
    )
    smoother = TemporalSmoother(include_mixed=include_mixed) if do_smooth else None

    import pandas as pd
    ok = 0
    errors = []
    for f in files:
        try:
            df = pd.read_csv(f)

            # Skip files that are empty after preprocessing (all trials excluded)
            if df.empty or len(df) == 0:
                logger.warning(f"Skipping {f.name}: empty after preprocessing")
                continue

            # ── sample-level classification ───────────────────────────────
            df = rec.infer_percept_from_closest_fixpoint(df)
            if include_mixed:
                df = rec.infer_percept_with_mixed(df)
            if use_ratio:
                df = rec.infer_percept_ratio(df)

            # Choose which percept column drives aggregation
            percept_col = "InferredPerceptRatio" if use_ratio else "InferredPercept"

            # ── section labelling & peri-jump exclusion ───────────────────
            df = rec.label_fixpoint_sections(df)
            if exclude_peri_jump:
                df = rec.exclude_peri_jump_samples(df, margin_ms=peri_jump_margin)

            # ── temporal smoothing (before aggregation) ───────────────────
            if smoother is not None:
                df = smoother.apply_median_filter(
                    df,
                    window_size=smooth_window,
                    percept_col=percept_col,
                    threshold=0.5,
                    delta=smooth_delta,
                ) if smooth_method == "median" else smoother.apply_uniform_filter(
                    df,
                    window_size=smooth_window,
                    percept_col=percept_col,
                    threshold=0.5,
                    delta=smooth_delta,
                )
                # Use the smoothed column for aggregation
                percept_col = f"{percept_col}Smoothed"

            # ── section-level aggregation ─────────────────────────────────
            if aggregate:
                df = rec.aggregate_by_fixation_section(
                    df,
                    percept_col=percept_col,
                    exclude_peri_jump=exclude_peri_jump,
                )

            # For report runs, join ground-truth percepts from perceptData CSV.
            # No-report runs have no perceptData file — silently skip.
            stem = re.sub(r"_preprocessed$", "", f.stem)
            is_report = f.stem.endswith("r_preprocessed")  # e.g. s12r03r_preprocessed
            if is_report:
                run_match = re.search(r"r(\d+)r_preprocessed", f.stem)
                if run_match:
                    run_nr = int(run_match.group(1))
                    try:
                        percept_data = load_percept_reports(subject_dir, run_nr)
                        df = join_reported_percepts(df, percept_data)
                        logger.info(f"Joined ReportedPercept for {f.name} (run {run_nr:02d})")
                    except FileNotFoundError:
                        logger.warning(
                            f"No perceptData file found for {f.name} run {run_nr:02d}; "
                            "ReportedPercept will not be added"
                        )

            out = percepts_dir / f"{stem}_percepts.csv"
            save_percept_data(df, out)

            # Evaluate on report runs when requested
            if do_evaluate and is_report and "ReportedPercept" in df.columns:
                try:
                    quality_dir = subject_dir / "quality"
                    evaluator = ReconstructionEvaluator()
                    results = evaluator.evaluate_run(df)
                    save_evaluation_results(results, quality_dir, stem)
                    s = results["summary"]
                    print(
                        f"  {stem}: acc={s['accuracy_2class']:.3f} "
                        f"F1={s['f1_macro_2class']:.3f} "
                        f"MCC={s['mcc_2class']:.3f} (2-class, section-level)"
                    )
                except Exception as e:
                    logger.warning(f"Evaluation failed for {stem}: {e}")

            logger.info(f"Reconstructed {f.name}")
            ok += 1
        except Exception as e:
            errors.append((f.name, str(e)))
            logger.error(f"Failed {f.name}: {e}")

    print(f"Reconstruction complete: {ok}/{len(files)} files → {percepts_dir}")
    if errors:
        print("Errors:", file=sys.stderr)
        for name, msg in errors:
            print(f"  {name}: {msg}", file=sys.stderr)


def run_process(args, config=None):
    """Run preprocess then reconstruct for a subject directory."""
    if config is None:
        config = load_config()
    run_preprocess(args, config)
    # reuse reconstruct with defaults
    args.include_mixed = True
    args.aggregate = True
    run_reconstruct(args, config)


def run_events(args, config=None):
    """Generate events.tsv and switch_events.tsv for all runs in a subject directory."""
    from bret.io.events import build_events_tsv, save_events_tsv

    if config is None:
        config = load_config()

    subject_dir = Path(args.subject_dir)
    percepts_dir = subject_dir / "percepts"
    if not percepts_dir.exists():
        print(
            f"ERROR: {percepts_dir} does not exist. "
            "Run 'bret reconstruct' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    files = sorted(percepts_dir.glob("*_percepts.csv"))
    if not files:
        print(f"No *_percepts.csv files found in {percepts_dir}", file=sys.stderr)
        sys.exit(1)

    recon_cfg = config.get("reconstruction", {})
    jump_window_ms = recon_cfg.get("jump_transition_window_ms", 200)
    smoothing_window_ms = recon_cfg.get("transition_smoothing_window_ms", 400)
    sampling_rate = config.get("preprocessing", {}).get("sampling_rate", 250)

    exclude_jump = getattr(args, "exclude_jump_induced", False)
    n_dummies = getattr(args, "n_dummies", 5)
    tr = getattr(args, "tr", 1.75)
    trial_dur = getattr(args, "trial_dur", 120.0)
    iti = getattr(args, "iti", 20.0)
    min_epoch_duration = getattr(args, "min_epoch_duration", None)

    events_dir = subject_dir / "events"
    events_dir.mkdir(exist_ok=True)

    ok, errors = 0, []
    for f in files:
        stem = f.stem.replace("_percepts", "")
        run_type = "no-report" if stem.endswith("nr") else "report"
        try:
            events_df, switch_df = build_events_tsv(
                subject_dir=subject_dir,
                stem=stem,
                run_type=run_type,
                exclude_jump_induced=exclude_jump,
                jump_window_ms=jump_window_ms,
                smoothing_window_ms=smoothing_window_ms,
                sampling_rate=sampling_rate,
                n_dummies=n_dummies,
                tr=tr,
                trial_dur=trial_dur,
                iti=iti,
                min_epoch_duration=min_epoch_duration,
            )
            out = events_dir / f"{stem}_events.tsv"
            switch_out = events_dir / f"{stem}_switch_events.tsv"
            save_events_tsv(events_df, out)
            save_events_tsv(switch_df, switch_out)
            print(f"  {stem}: {len(events_df)} epochs → {out.relative_to(subject_dir.parent)}")
            ok += 1
        except FileNotFoundError as e:
            # Training/gammafit report runs have no perceptData — expected, not an error
            logger.info(f"Skipping {stem}: {e}")
        except Exception as e:
            errors.append((stem, str(e)))
            logger.error(f"Events failed for {stem}: {e}")

    print(f"Events complete: {ok}/{len(files)} runs → {events_dir}")
    if errors:
        print("Errors:", file=sys.stderr)
        for stem, msg in errors:
            print(f"  {stem}: {msg}", file=sys.stderr)


def run_validate(args, config):
    """Execute validation command."""
    logger.info(f"Validating data for subject {args.subject}")
    raise NotImplementedError("Validate command not yet implemented")


def run_batch(args, config=None):
    """Run preprocess + reconstruct for multiple subjects."""
    if config is None:
        config = load_config()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory {data_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.subjects:
        subject_nums = [s.strip() for s in args.subjects.split(",")]
        subject_dirs = [data_dir / f"sub-{n}" for n in subject_nums]
    else:
        subject_dirs = sorted(data_dir.glob("sub-*"))

    subject_dirs = [d for d in subject_dirs if d.is_dir()]
    if not subject_dirs:
        print("No subject directories found.", file=sys.stderr)
        sys.exit(1)

    print(f"Batch processing {len(subject_dirs)} subject(s): "
          f"{[d.name for d in subject_dirs]}")

    import types
    ok, errors = 0, []
    for subject_dir in subject_dirs:
        sub_args = types.SimpleNamespace(
            subject_dir=str(subject_dir),
            config=getattr(args, "config", None),
            include_mixed=True,
            aggregate=True,
            evaluate=getattr(args, "evaluate", False),
        )
        try:
            if not args.reconstruct_only:
                run_preprocess(sub_args, config)
            if not args.preprocess_only:
                run_reconstruct(sub_args, config)
            if getattr(args, "events", False):
                sub_args.exclude_jump_induced = getattr(args, "exclude_jump_induced", False)
                sub_args.n_dummies = getattr(args, "n_dummies", 5)
                sub_args.tr = getattr(args, "tr", 1.75)
                sub_args.trial_dur = getattr(args, "trial_dur", 120.0)
                sub_args.iti = getattr(args, "iti", 20.0)
                run_events(sub_args, config)
            ok += 1
        except SystemExit:
            errors.append(subject_dir.name)
        except Exception as e:
            errors.append(subject_dir.name)
            logger.error(f"Failed {subject_dir.name}: {e}")

    print(f"\nBatch complete: {ok}/{len(subject_dirs)} subjects succeeded.")
    if errors:
        print(f"Failed: {errors}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
