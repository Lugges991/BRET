# BRET — Binocular Rivalry Eye Tracking

A Python package for preprocessing EyeLink eye-tracking data and reconstructing
perceptual states from binocular rivalry experiments, with optional BIDS-compatible
events-TSV output for fMRI analysis.

---

## Overview

In binocular rivalry experiments a different image (e.g. house vs. face) is
presented to each eye.  Conscious perception alternates spontaneously between
the two.  BRET infers which percept is dominant at every moment from **where
the participant looks** relative to a set of jumping fixation points — no
button-press required.

```
Raw .asc  →  preprocessing  →  features  →  reconstruction  →  percepts CSV
                                                    ↓
                                             evaluators  →  quality JSON/CSV
                                                    ↓
                                             events.py  →  *_events.tsv (BIDS)
```

Two paradigms are supported:

| Run type | Ground truth | Evaluation |
|---|---|---|
| **report** | Button-press percept markers | ✅ section-level F1 / MCC |
| **no-report** | None (pure gaze inference) | — |

---

## Installation

```bash
git clone https://github.com/yourusername/BRET.git
cd BRET
pip install -e ".[dev]"
```

**Requirements**: Python ≥ 3.8, NumPy, Pandas, SciPy, scikit-learn, matplotlib,
seaborn, PyYAML. See [requirements.txt](requirements.txt).

---

## Quick Start

### Single subject — full pipeline

```bash
bret process --subject-dir data/sub-11
```

### Batch — all subjects

```bash
bret batch --data-dir data                        # preprocess + reconstruct
bret batch --data-dir data --evaluate             # + quality eval on report runs
bret batch --data-dir data --events               # + generate events.tsv for fMRI
bret batch --data-dir data --subjects 11,12,13    # specific subjects only
bret batch --data-dir data --reconstruct-only     # skip preprocessing
```

### Generate BIDS events TSV

```bash
bret events --subject-dir data/sub-11
```

This creates `data/sub-11/events/{stem}_events.tsv` and `{stem}_switch_events.tsv`
for every reconstructed run.  Scanner timing defaults can be overridden:

```bash
bret events --subject-dir data/sub-11 \
    --n-dummies 5 --tr 1.75 --trial-dur 120 --iti 20
```

Onset formula: `onset = n_dummies × TR + ITI + (ITI + trial_dur) × (N − 1) + epoch_onset`

### Programmatic usage

```python
from bret.preprocessing.pipeline import PreprocessingPipeline
from bret.reconstruction.euclidean import EuclideanReconstructor
from bret.io.loaders import load_preprocessed_data
from bret.utils.config_loader import load_config
from pathlib import Path

config = load_config()

# Preprocess
pipeline = PreprocessingPipeline(config)
pipeline.process_file(Path("data/sub-11/s11r06r.asc"),
                      output_file=Path("data/sub-11/preprocessed/s11r06r_preprocessed.csv"))

# Reconstruct percepts
df = load_preprocessed_data("data/sub-11/preprocessed/s11r06r_preprocessed.csv")
rec = EuclideanReconstructor(include_mixed=True)
df = rec.infer_percept_from_closest_fixpoint(df)   # adds InferredPercept
df = rec.label_fixpoint_sections(df)               # adds FixpointSection
df = rec.aggregate_by_fixation_section(df)         # adds InferredPerceptAggregated
```

---

## File Naming Conventions

| File | Pattern | Location |
|---|---|---|
| Raw eye-tracker | `s{N}r{RR}{r\|nr}.asc` | `data/sub-{N}/` |
| Preprocessed | `s{N}r{RR}{r\|nr}_preprocessed.csv` | `data/sub-{N}/preprocessed/` |
| Percepts | `s{N}r{RR}{r\|nr}_percepts.csv` | `data/sub-{N}/percepts/` |
| Events TSV | `s{N}r{RR}{r\|nr}_events.tsv` | `data/sub-{N}/events/` |
| Switch events | `s{N}r{RR}{r\|nr}_switch_events.tsv` | `data/sub-{N}/events/` |
| Eval summary | `s{N}r{RR}r_eval_summary.json` | `data/sub-{N}/quality/` |

`r` = report run, `nr` = no-report run.

---

## Configuration

All parameters live in [bret/config/default_config.yaml](bret/config/default_config.yaml).

Key sections:

| Section | Key parameter | Default |
|---|---|---|
| `preprocessing` | `lp` (Butterworth cutoff Hz) | 30 |
| `preprocessing` | `missing_data_threshold` | 0.30 |
| `reconstruction` | `include_mixed` | true |
| `reconstruction` | `use_ratio_classifier` | false |
| `reconstruction` | `ratio_threshold` τ | 0.0 |
| `reconstruction` | `peri_jump_margin_ms` | 50 |
| `reconstruction` | `apply_smoothing` | false |
| `reconstruction` | `smoothing_window_ms` | 150 |
| `screen` | pixel→degree conversion params | — |

Pass a custom config file with `--config path/to/config.yaml`.

---

## Module Status

| Module | Status | Notes |
|---|---|---|
| `preprocessing/` | ✅ complete | Parse → clean → filter → align → validate |
| `features/spatial.py` | ✅ complete | Euclidean distance to fixpoints in deg |
| `features/temporal.py` | ✅ `label_percept_transitions()` | `compute_fixation_duration()`, `compute_rolling_majority()` still stubbed |
| `features/motion.py` | ⚠️ stubs | `compute_velocity()`, `compute_acceleration()` raise `NotImplementedError` |
| `reconstruction/euclidean.py` | ✅ complete | 2-class, 3-class, log-ratio, section aggregation |
| `reconstruction/smoothing.py` | ✅ complete | Uniform/median filter, hysteresis, grid search |
| `reconstruction/evaluators.py` | ✅ complete | Section / trial / run-level F1 & MCC |
| `io/` | ✅ complete | Loaders, writers, events TSV builder |
| `cli/` | ✅ mostly complete | `validate` subcommand raises `NotImplementedError` |
| `quality/` | ✅ complete | File validation, QC reports |
| `visualization/` | ❌ stubs | `gaze_plots`, `percept_plots`, `quality_plots` |

---

## Classification Performance

Evaluated on report runs (section-level modal percept vs. inferred percept):

| Metric | Value |
|---|---|
| Mean 2-class accuracy | 0.799 |
| Median 2-class accuracy | 0.836 |
| Runs below 0.70 | 9 / 32 |
| Worst subjects | sub-15 (0.58), sub-14 (0.62), sub-18 (0.64) |

Full cross-subject report: `data/eval_report_all_subjects.csv`.

---

## Data Coverage

| Subject | Runs | Preprocessed | Percepts | Quality evals | Notes |
|---|---|---|---|---|---|
| sub-11 | 9 | 9 | 9 | 12 | Reference dataset |
| sub-12 | 10 | 10 | 10 | 12 | |
| sub-13 | 10 | 10 | 10 | 12 | |
| sub-14 | 11 | 11 | 11 | 12 | 2 training runs |
| sub-15 | 12 | 12 | 12 | 12 | run06 flagged incomplete |
| sub-16 | 10 | 10 | 10 | 12 | |
| sub-17 | 11 | 0 | 0 | 0 | **Preprocessing not yet run** |
| sub-18 | 10 | 10 | 10 | 12 | |
| sub-19 | 8 | 8 | 8 | 9 | Fewest runs |
| sub-20 | 10 | 10 | 10 | 12 | Most recent (Jan 2026) |
| **Total** | **101** | **90** | **90** | **105** | 39 report + 41 no-report rivalry |

---

## Roadmap / To-Do

### High priority

- [ ] **Process sub-17** — only unprocessed subject; 11 `.asc` files ready
- [ ] **Improve classification for sub-14/15/18** — tune `ratio_threshold` and
  `smoothing_window_ms` per-subject via `bret batch --evaluate` + `grid_search_optimal_window()`
- [ ] **Label fixation-jump-induced vs. spontaneous transitions** (`features/temporal.py::label_percept_transitions()`)
  — compare fraction of jump-induced switches across run types; test whether excluding them improves evaluation
- [ ] **`bret validate` CLI** — currently raises `NotImplementedError`

### Medium priority

- [ ] **Visualization** — implement `plot_gaze_trajectory()`, `plot_percept_timeline()`, `plot_preprocessing_quality()` in `visualization/`
- [ ] **`compute_fixation_duration()` / `compute_rolling_majority()`** in `features/temporal.py`
- [ ] **`compute_velocity()` / `compute_acceleration()`** in `features/motion.py`
- [ ] **Per-subject tuning report** — automate grid-search + best-params summary for each subject
- [ ] **`coalesce_events()`** and **`check_missing_data_threshold()`** stubs in preprocessing

### Lower priority / Future

- [ ] **BIDS full conversion** — sidecar JSON, coordsystem, dataset_description
- [ ] **ML classifier** — replace Euclidean heuristic with trained model (deliberately out of scope for this package; see separate repo)
- [ ] **Replay trial accuracy** — evaluate reconstruction quality specifically on replay trials
- [ ] Sub-15 run06 incomplete data — decide on exclusion vs. partial use
- [ ] Handle sub-09 reversed key assignments (house/face flipped — see deprecated code)

---

## Testing

```bash
pytest tests/
python test_process_file.py    # quick integration check (preprocessing)
python test_reconstruct.py     # quick integration check (reconstruction)
```

Reference dataset for regression testing: `data/sub-11/` (complete training →
gammafit → report → no-report sequence).

---

## Project Structure

```
bret/
├── cli/            Entry points (bret preprocess / reconstruct / process / batch / events)
├── config/         default_config.yaml
├── features/       spatial.py ✅  temporal.py ⚠️  motion.py ❌
├── io/             loaders.py  writers.py  events.py  parsers.py
├── preprocessing/  parser → cleaning → filtering → alignment → validation → pipeline
├── quality/        checks.py  reports.py
├── reconstruction/ euclidean.py  smoothing.py  evaluators.py
├── utils/          config_loader  screen_params  metrics  logging_setup
└── visualization/  gaze_plots ❌  percept_plots ❌  quality_plots ❌
docs/
├── METHODS.md      Experimental methods skeleton
└── classification_improvements.md
examples/
    batch_process_all_subjects.py
    compare_configs.py
    generate_eval_report.py
    saccade_induced_switches.py
    ...
```

---

## Citation / Acknowledgements

If you use this code, please cite the associated paper (in preparation).

---

## License

[To be determined]
