"""
Data cleaning functions for eye tracking data.

Handles blink interpolation, outlier detection, and event coalescing.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def interpolate_blinks(
    df: pd.DataFrame,
    diameter_threshold,
    coalesce,
    blink_padding,
    saccade_padding,
    sampling_rate,
    screen_resolution,
    x_offset,
    y_offset,
    ) -> pd.DataFrame:
    """
    Interpolate missing data during blinks and saccades.
    
    Args:
        df: DataFrame with eye tracking data
        blink_padding: Time to pad around blinks (seconds)
        saccade_padding: Time to pad around saccades (seconds)
        diameter_threshold: Pupil diameter threshold for blink detection
        
    Returns:
        DataFrame with interpolated values
    """
    logger.info("Interpolating blinks and saccades")
    
    # 1. Detect undetected blinks (diameter <= threshold)
    df_interpolated = df.copy()
    df_interpolated.loc[df_interpolated['Diameter'] <= diameter_threshold, 'Blink'] = 1

    # 2. Coalesce blinks within threshold - optimized with NumPy arrays
    blink_indices = df_interpolated.index[df_interpolated['Blink'] == 1].tolist()
    if len(blink_indices) > 1:
        timestamps = df_interpolated['Timestamp'].values
        blink_mask = df_interpolated['Blink'].values.copy()
        
        for i in range(1, len(blink_indices)):
            if timestamps[blink_indices[i]] - timestamps[blink_indices[i-1]] < coalesce:
                blink_mask[blink_indices[i-1]:blink_indices[i]] = 1
        
        df_interpolated['Blink'] = blink_mask
    
    # 3. Pad the blinks - optimized with vectorized operations
    blink_starts = df_interpolated[(df_interpolated['Blink'] == 1) & (df_interpolated['Blink'].shift(1) == 0)].index
    blink_ends = df_interpolated[(df_interpolated['Blink'] == 1) & (df_interpolated['Blink'].shift(-1) == 0)].index
    
    # Use NumPy array for faster padding operations
    blink_mask = df_interpolated['Blink'].values.copy()
    padding_samples = int(blink_padding * sampling_rate)
    
    for idx in blink_starts:
        start = max(0, idx - padding_samples)
        blink_mask[start:idx] = 1
    
    for idx in blink_ends:
        end = min(len(blink_mask), idx + padding_samples + 1)
        blink_mask[idx:end] = 1
    
    df_interpolated['Blink'] = blink_mask
    
    # Set all the values in the X, Y and Diameter columns to NaN where the Blink column contains 1
    blink_mask_bool = blink_mask == 1
    df_interpolated.loc[blink_mask_bool, ['X', 'Y', 'Diameter']] = np.nan

    # 4. Pad saccades and set to NaN - optimized with vectorized operations
    saccade_starts = df_interpolated[(df_interpolated['Saccade'] == 1) & (df_interpolated['Saccade'].shift(1) == 0)].index
    saccade_ends = df_interpolated[(df_interpolated['Saccade'] == 1) & (df_interpolated['Saccade'].shift(-1) == 0)].index
    
    # Use NumPy array for faster padding operations
    saccade_mask = df_interpolated['Saccade'].values.copy()
    padding_samples = int(saccade_padding * sampling_rate)
    
    for idx in saccade_starts:
        start = max(0, idx - padding_samples)
        saccade_mask[start:idx] = 1
    
    for idx in saccade_ends:
        end = min(len(saccade_mask), idx + padding_samples + 1)
        saccade_mask[idx:end] = 1
    
    df_interpolated['Saccade'] = saccade_mask
    df_interpolated.loc[saccade_mask == 1, ['X', 'Y', 'Diameter']] = np.nan

    # 5. Find values beyond the screen and set them to NaN - optimized with NumPy arrays
    screen_left = 0 - x_offset
    screen_right = screen_resolution[0] - float(x_offset)
    screen_top = 0 - y_offset
    screen_bottom = screen_resolution[1] - float(y_offset)
    
    # Use NumPy arrays for faster boundary checking (must copy to avoid read-only issues)
    x_vals = df_interpolated['X'].values.copy()
    y_vals = df_interpolated['Y'].values.copy()
    
    x_vals[(x_vals < screen_left) | (x_vals > screen_right)] = np.nan
    y_vals[(y_vals < screen_top) | (y_vals > screen_bottom)] = np.nan
    
    df_interpolated['X'] = x_vals
    df_interpolated['Y'] = y_vals
    
    # 6. Detect extreme values for diameter and set to NaN (median +/- 3 std) - optimized
    diameter_vals = df_interpolated['Diameter'].values.copy()
    diameter_median = np.nanmedian(diameter_vals)
    diameter_std = np.nanstd(diameter_vals)
    
    diameter_vals[(diameter_vals < (diameter_median - 3 * diameter_std)) | 
                  (diameter_vals > (diameter_median + 3 * diameter_std))] = np.nan
    
    df_interpolated['Diameter'] = diameter_vals

    # 7. Interpolate NaN values - combined for efficiency
    df_interpolated[['X', 'Y', 'Diameter']] = df_interpolated[['X', 'Y', 'Diameter']].interpolate(
        method='linear', limit_direction='both'
    )

    return df_interpolated


def detect_outliers(
    df: pd.DataFrame,
    missing_data_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Detect outliers using median absolute deviation.
    
    Args:
        df: DataFrame with data
        columns: Columns to check for outliers (default: ['X', 'Y', 'Diameter'])
        n_std: Number of standard deviations for outlier threshold
        
    Returns:
        DataFrame with outliers marked as NaN
    """
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


def coalesce_events(
    df: pd.DataFrame,
    event_column: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Coalesce nearby events within threshold duration.
    
    Args:
        df: DataFrame with event data
        event_column: Name of the event column (e.g., 'Blink')
        threshold: Time threshold for coalescing (seconds)
        
    Returns:
        DataFrame with coalesced events
    """
    # TODO: Implement event coalescing
    raise NotImplementedError("Event coalescing not yet implemented")
