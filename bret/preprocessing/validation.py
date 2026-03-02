"""
Data quality validation functions.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def check_missing_data_threshold(
    df: pd.DataFrame,
    threshold: float = 0.3,
    columns: list = None,
) -> bool:
    """
    Check if missing data exceeds threshold.
    
    Args:
        df: DataFrame to check
        threshold: Maximum allowed proportion of missing data
        columns: Columns to check (default: ['X', 'Y'])
        
    Returns:
        True if data quality is acceptable, False otherwise
    """
    if columns is None:
        columns = ['X', 'Y']
    
    # TODO: Implement missing data check
    raise NotImplementedError("Missing data check not yet implemented")


def exclude_low_quality_trials(
    df: pd.DataFrame,
    missing_data_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Exclude trials with excessive missing data.
    
    Args:
        df: DataFrame with trial data
        threshold: Maximum allowed proportion of missing data per trial
        
    Returns:
        DataFrame with low-quality trials removed
    """
    logger.info(f"Excluding trials with >{missing_data_threshold*100}% missing data")
    
    # Calculate the percentage of missing data for each trial (= number of columns with blink values set to 1)
    trial_lengths = df['Trial'].value_counts()
    missing_data = df.groupby('Trial')['Blink'].sum() / trial_lengths
    #print(f'The percentage of missing data for each trial is: {missing_data}')

    # Exclude trials in which more than 30% of the data is missing
    trials_to_exclude = missing_data[missing_data > missing_data_threshold].index
    df = df[~df['Trial'].isin(trials_to_exclude)]

    # Print names of trials that were excluded
    if len(trials_to_exclude) > 0:
        logger.info(f'The following trials were excluded because more than {missing_data_threshold*100}% of the data was missing: {trials_to_exclude.tolist()}')
    else:
        logger.info(f'No trials were excluded because more than {missing_data_threshold*100}% of the data was missing.') 
    
    return df

def extract_trial_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract trial data from parsed DataFrame.
    
    Args:
        df: Parsed DataFrame with raw data
    Returns:
        DataFrame with trial data extracted
    """


    # Run-relative timestamp (seconds from first sample of the run).
    # Computed before trial-relativising so it is never reset to 0.
    run_start = df["Timestamp"].min()
    df["TimestampRun"] = df["Timestamp"] - run_start

    # drop all rows where Trial is NaN
    df = df.dropna(subset=['Fixpoint1'])
    # Trial-relative timestamp (seconds from the start of each trial).
    df["Timestamp"] = df["Timestamp"] - df.groupby("Trial")["Timestamp"].transform("min")
    return df
