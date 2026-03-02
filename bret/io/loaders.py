"""
Data loading functions.
"""

import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from typing import Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


def load_subject_preprocessed_data(
    sub_dir: Path,
) -> pd.DataFrame:
    """
    Load preprocessed eye tracking data.
    
    Args:
        sub_dir: Path to subject directory
    Returns:
        DataFrame with preprocessed data
    """

    preprocessed_dir = sub_dir / "preprocessed"

    preprocessed_files = list(sorted(preprocessed_dir.glob("*_preprocessed.csv")))

    # get sub and run nr from file
    # s20r10nr_preprocessed.csv -> sub-20, run-10
    dfs = []

    for file in preprocessed_files:
        sub = re.search(r"s(\d+)", file.stem).group(1)
        run = re.search(r"r(\d+)(?:nr|r)", file.stem).group(1)

        #  trial type n or r -> report or no report
        run_type = "report" if "r" in file.stem and "nr" not in file.stem else "no report"
        
        df = pd.read_csv(file)

        df['Subject'] = int(sub)
        df['Run'] = int(run)
        df['RunType'] = run_type
        dfs.append(df)
    return dfs



def load_percept_reports(
    subject_dir: Path,
    run_nr: int,
) -> pd.DataFrame:
    """
    Load percept report data from a MATLAB-generated perceptData CSV file.

    Searches for ``sub-{N}_run{RR}_report_perceptData_*.csv`` inside
    *subject_dir*.  If multiple files match (e.g. re-runs), the lexicographically
    last one (most recent timestamp suffix) is used.

    Args:
        subject_dir: Subject data directory (e.g. ``data/sub-12``).
        run_nr: Integer run number (e.g. 3 for run03).

    Returns:
        DataFrame with columns: trial, type, leftImage, percept, onset, duration.

    Raises:
        FileNotFoundError: If no matching perceptData file is found.
    """
    subject_dir = Path(subject_dir)
    sub_match = re.search(r"sub-(\d+)", subject_dir.name)
    sub_nr = sub_match.group(1) if sub_match else "*"

    pattern = f"sub-{sub_nr}_run{run_nr:02d}_report_perceptData_*.csv"
    matches = sorted(subject_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No perceptData file found matching '{pattern}' in {subject_dir}"
        )
    if len(matches) > 1:
        logger.warning(
            f"Multiple perceptData files matched '{pattern}'; "
            f"using most recent: {matches[-1].name}"
        )

    path = matches[-1]
    df = pd.read_csv(path)
    logger.info(f"Loaded percept reports ({len(df)} epochs) from {path.name}")
    return df


def join_reported_percepts(
    df: pd.DataFrame,
    percept_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a ``ReportedPercept`` column to *df* from perceptData changepoint records.

    For each sample row the function looks up which perceptData epoch
    (onset <= Timestamp < onset + duration) within the same trial it belongs to
    and assigns that epoch's percept label.  Samples that fall outside any
    epoch (gaps between button presses) receive ``pd.NA``.

    The .asc-derived ``Percept`` column is left intact so the caller can
    verify or compare; evaluation code should use ``ReportedPercept``.

    Args:
        df: Sample-level DataFrame with at minimum ``Timestamp`` and ``Trial``.
        percept_data: Changepoint DataFrame as returned by
            :func:`load_percept_reports` (columns: trial, percept, onset, duration).

    Returns:
        Copy of *df* with an added ``ReportedPercept`` column.
    """
    percept_data = percept_data.copy()
    percept_data["_end"] = percept_data["onset"] + percept_data["duration"]

    if df.empty:
        logger.warning("join_reported_percepts: input DataFrame is empty, returning as-is")
        df = df.copy()
        df["ReportedPercept"] = pd.NA
        return df

    result_parts = []
    for trial_id, trial_df in df.groupby("Trial", sort=False):
        pd_trial = (
            percept_data[percept_data["trial"] == int(trial_id)]
            .sort_values("onset")
            .reset_index(drop=True)
        )
        trial_df = trial_df.copy()

        if pd_trial.empty:
            logger.warning(f"No perceptData epochs for trial {trial_id}; ReportedPercept will be NA")
            trial_df["ReportedPercept"] = pd.NA
            result_parts.append(trial_df)
            continue

        # merge_asof: for each Timestamp find the epoch whose onset is <= Timestamp
        merged = pd.merge_asof(
            trial_df.sort_values("Timestamp"),
            pd_trial[["onset", "_end", "percept"]].rename(
                columns={"percept": "ReportedPercept"}
            ),
            left_on="Timestamp",
            right_on="onset",
            direction="backward",
        )
        # Null out samples that fall past the epoch's end (inter-press gaps)
        out_of_epoch = merged["Timestamp"] >= merged["_end"]
        if out_of_epoch.any():
            logger.debug(
                f"Trial {trial_id}: {out_of_epoch.sum()} samples outside any epoch → NA"
            )
        merged.loc[out_of_epoch, "ReportedPercept"] = pd.NA
        merged = merged.drop(columns=["onset", "_end"])

        result_parts.append(merged)

    out = pd.concat(result_parts).sort_index()
    n_labelled = out["ReportedPercept"].notna().sum()
    n_total = len(out)
    logger.info(
        f"Joined reported percepts: {n_labelled}/{n_total} samples labelled "
        f"({100*n_labelled/n_total:.1f}%)"
    )
    return out



def load_percept_sidecar(
    sidecar_path: Path,
    preprocessed_path: Path = None,
) -> pd.DataFrame:
    """
    Load a percept sidecar CSV, optionally joined with its preprocessed file.

    The sidecar contains only the reconstruction-derived columns
    (``Timestamp``, ``Trial``, ``InferredPercept``, ``InferredPerceptMixed``,
    ``FixpointSection``, ``InferredPerceptAggregated``, ``ReportedPercept``).
    Passing *preprocessed_path* merges in all raw gaze columns so the caller
    gets a single complete DataFrame.

    Args:
        sidecar_path: Path to ``*_percepts.csv`` sidecar file.
        preprocessed_path: Optional path to the matching ``*_preprocessed.csv``.
            When given, the two files are merged on ``['Timestamp', 'Trial']``.

    Returns:
        Sidecar DataFrame, or merged full DataFrame if *preprocessed_path* given.
    """
    sidecar_path = Path(sidecar_path)
    sidecar = pd.read_csv(sidecar_path)
    logger.info(f"Loaded percept sidecar ({len(sidecar)} rows, {list(sidecar.columns)}) from {sidecar_path.name}")

    if preprocessed_path is None:
        return sidecar

    preprocessed_path = Path(preprocessed_path)
    pre = pd.read_csv(preprocessed_path)
    logger.info(f"Merging with preprocessed file {preprocessed_path.name}")
    merged = pre.merge(sidecar, on=["Timestamp", "Trial"], how="left")
    return merged


def load_replay_data(
    filepath: Path,
) -> pd.DataFrame:
    """
    Load replay trial data.
    
    Args:
        filepath: Path to replay CSV file
        
    Returns:
        DataFrame with replay data
    """
    # TODO: Implement replay data loading
    raise NotImplementedError("Load replay data not yet implemented")


def load_mat_offsets(
    filepath: Path,
) -> Dict[str, float]:
    """
    Load screen offset values from .mat file.
    
    Args:
        filepath: Path to stimuli .mat file
        
    Returns:
        Dictionary with 'xOffset' and 'yOffset' keys
    """
    # TODO: Implement .mat loading using scipy.io.loadmat
    raise NotImplementedError("Load MAT offsets not yet implemented")
