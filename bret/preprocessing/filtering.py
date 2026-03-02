"""
Signal filtering functions for eye tracking data.
"""

import pandas as pd
import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


def apply_butterworth_filter(
    df: pd.DataFrame,
    order: int = 4,
    cutoff: float = 30.0,
    sampling_rate: float = 250.0,
) -> pd.DataFrame:
    """
    Apply Butterworth low-pass filter to gaze data X and Y columns.
    
    Args:
        df: DataFrame with eye tracking data
        cutoff: Cutoff frequency in Hz
        order: Filter order
        sampling_rate: Sampling rate in Hz
        
    Returns:
        DataFrame with filtered data
    """
    logger.info(f"Applying Butterworth filter (cutoff={cutoff}Hz, order={order})")
    df_filtered = df.copy()

    
    # Apply low-pass filter
    b, a = signal.butter(order, cutoff / (sampling_rate / 2), btype='low')
    df_filtered['X'] = signal.filtfilt(b, a, df_filtered['X'])
    df_filtered['Y'] = signal.filtfilt(b, a, df_filtered['Y'])
  
    return df
    
