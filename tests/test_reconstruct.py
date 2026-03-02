"""
Integration test for the reconstruction pipeline (Goal A).

Tests EuclideanReconstructor + save_percept_data against the reference
preprocessed file data/sub-11/preprocessed/s11r04r_preprocessed.csv.

Run with:
    python test_reconstruct.py
"""

import sys
import traceback
import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/sub-11/preprocessed/s11r04r_preprocessed.csv")
PASSES = []
FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        PASSES.append(name)
        print(f"  PASS  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── 0. prerequisites ──────────────────────────────────────────────────────────
section("0. Prerequisites")

if not INPUT.exists():
    print(f"  ERROR: Input file not found: {INPUT}")
    print("  Run 'bret preprocess --subject-dir data/sub-11' first.")
    sys.exit(1)

df_raw = pd.read_csv(INPUT)
print(f"  Loaded {len(df_raw):,} rows from {INPUT.name}")

required_cols = {"Timestamp", "Trial", "Type", "Image1",
                 "Fixpoint1", "Fixpoint2",
                 "DistanceToFixpoint1", "DistanceToFixpoint2", "DistanceToCenter"}
missing = required_cols - set(df_raw.columns)
check("required columns present", not missing, f"missing: {missing}")

# ── 1. infer_percept_from_closest_fixpoint ────────────────────────────────────
section("1. EuclideanReconstructor.infer_percept_from_closest_fixpoint()")

from bret.reconstruction.euclidean import EuclideanReconstructor

rec = EuclideanReconstructor(include_mixed=True)
df = rec.infer_percept_from_closest_fixpoint(df_raw.copy())

check("InferredPercept column added",
      "InferredPercept" in df.columns)
check("InferredPercept has no NaNs",
      df["InferredPercept"].notna().all())
check("InferredPercept values are house or face",
      set(df["InferredPercept"].unique()).issubset({"house", "face"}))

# Verify logic: when dist1 < dist2 → should equal Image1
mask = df["DistanceToFixpoint1"] < df["DistanceToFixpoint2"]
correct = (df.loc[mask, "InferredPercept"] == df.loc[mask, "Image1"]).all()
check("percept == Image1 when dist1 < dist2", correct)

# ── 2. infer_percept_with_mixed ───────────────────────────────────────────────
section("2. EuclideanReconstructor.infer_percept_with_mixed()")

df = rec.infer_percept_with_mixed(df)

check("'InferredPerceptMixed' column added",
      "InferredPerceptMixed" in df.columns)
check("mixed/house/face only",
      set(df["InferredPerceptMixed"].unique()).issubset({"house", "face", "mixed"}))

# All rows where center is closest should be 'mixed'
center_closest = (
    (df["DistanceToCenter"] < df["DistanceToFixpoint1"]) &
    (df["DistanceToCenter"] < df["DistanceToFixpoint2"])
)
check("center-closest rows classified as mixed",
      (df.loc[center_closest, "InferredPerceptMixed"] == "mixed").all())

mixed_frac = center_closest.mean()
check("mixed fraction is plausible (0–50%)",
      0.0 <= mixed_frac <= 0.5,
      f"mixed fraction = {mixed_frac:.3f}")
print(f"         mixed fraction: {mixed_frac:.3f}")

# ── 3. aggregate_by_fixation_section ─────────────────────────────────────────
section("3. EuclideanReconstructor.aggregate_by_fixation_section()")

df = rec.aggregate_by_fixation_section(df, percept_col="InferredPercept", method="mode")

check("FixpointSection column added",
      "FixpointSection" in df.columns)
check("InferredPerceptAggregated column added",
      "InferredPerceptAggregated" in df.columns)
check("FixpointSection starts at 1",
      df["FixpointSection"].min() == 1)
check("FixpointSection is strictly monotone non-decreasing",
      (df["FixpointSection"].diff().dropna() >= 0).all())

n_sections = df["FixpointSection"].nunique()
check("plausible number of sections (10–500)",
      10 <= n_sections <= 500,
      f"got {n_sections}")
print(f"         sections detected: {n_sections}")

# Section boundaries align with fixpoint changes
boundaries = df[df["FixpointSection"] != df["FixpointSection"].shift()].index[1:]
fp1_changed = (df.loc[boundaries, "Fixpoint1"].ffill().values !=
               df.loc[boundaries - 1, "Fixpoint1"].ffill().values)
fp2_changed = (df.loc[boundaries, "Fixpoint2"].ffill().values !=
               df.loc[boundaries - 1, "Fixpoint2"].ffill().values)
check("every section boundary coincides with a fixpoint change",
      (fp1_changed | fp2_changed).all())

check("InferredPerceptAggregated values are house or face",
      set(df["InferredPerceptAggregated"].dropna().unique()).issubset({"house", "face"}))

# ── 4. save_percept_data — timestamp format ───────────────────────────────────
section("4. save_percept_data(format_type='timestamp')")

from bret.io.writers import save_percept_data
import tempfile, os

with tempfile.TemporaryDirectory() as tmp:
    ts_path = Path(tmp) / "test_timestamp.csv"
    save_percept_data(df, ts_path, percept_col="InferredPercept", format_type="timestamp")

    check("timestamp file created", ts_path.exists())
    ts = pd.read_csv(ts_path)
    check("timestamp rows match input", len(ts) == len(df))
    check("InferredPercept column present in output", "InferredPercept" in ts.columns)
    check("FixpointSection column present in output", "FixpointSection" in ts.columns)

# ── 5. save_percept_data — changepoint format ─────────────────────────────────
section("5. save_percept_data(format_type='changepoint')")

with tempfile.TemporaryDirectory() as tmp:
    cp_path = Path(tmp) / "test_changepoint.csv"
    save_percept_data(df, cp_path, percept_col="InferredPercept", format_type="changepoint")

    check("changepoint file created", cp_path.exists())
    cp = pd.read_csv(cp_path)

    expected_cols = {"trial", "type", "leftImage", "percept", "onset", "duration"}
    check("changepoint has correct columns",
          expected_cols.issubset(set(cp.columns)),
          f"missing: {expected_cols - set(cp.columns)}")
    check("fewer changepoint rows than input rows", len(cp) < len(df))
    check("all durations are non-negative", (cp["duration"] >= 0).all())
    check("percept values are house/face only",
          set(cp["percept"].unique()).issubset({"house", "face"}))
    print(f"         changepoint epochs: {len(cp):,}")
    print(f"         median epoch duration: {cp['duration'].median():.3f} s")

# ── 6. error handling ─────────────────────────────────────────────────────────
section("6. Error handling")

try:
    rec.aggregate_by_fixation_section(df_raw.copy(), percept_col="NonExistentCol")
    check("raises ValueError for missing percept_col", False)
except ValueError:
    check("raises ValueError for missing percept_col", True)

try:
    save_percept_data(df, Path("/tmp/dummy.csv"), percept_col="NonExistentCol")
    check("raises ValueError for missing save col", False)
except ValueError:
    check("raises ValueError for missing save col", True)

try:
    save_percept_data(df, Path("/tmp/dummy.csv"), percept_col="InferredPercept",
                      format_type="bad_format")
    check("raises ValueError for unknown format_type", False)
except ValueError:
    check("raises ValueError for unknown format_type", True)

# ── Summary ───────────────────────────────────────────────────────────────────
section("Summary")
total = len(PASSES) + len(FAILURES)
print(f"  {len(PASSES)}/{total} checks passed")
if FAILURES:
    print(f"\n  Failed checks:")
    for f in FAILURES:
        print(f"    • {f}")
    sys.exit(1)
else:
    print("  All checks passed.")
