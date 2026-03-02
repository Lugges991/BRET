"""
Functions for aligning gaze data to screen coordinates.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_centroid(df: pd.DataFrame, threshold) -> Tuple[float, float]:
    """
    Calculate the centroid of gaze positions.
    
    Args:
        df: DataFrame with X and Y coordinates
        
    Returns:
        Tuple of (x_centroid, y_centroid)
    """
    # TODO: Implement centroid calculation
    # Calculate the centroid of the gaze data
    x = df['X']
    y = df['Y']
    centroid = [x.mean(), y.mean()]

    # Calculate the distance of each point to the centroid
    distances = np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2)
    
    # Remove outliers
    data = df[distances < threshold]
    
    # Calculate the centroid again
    x = data['X']
    y = data['Y']
    return (x.mean(), y.mean())

def align_to_center(
    df: pd.DataFrame,
    threshold,
    screen_center: Tuple[float, float] = None,
) -> pd.DataFrame:
    """
    Align gaze coordinates to screen center.
    
    Args:
        df: DataFrame with X and Y coordinates
        screen_center: Center coordinates (if None, calculated from data)
        use_mat_offsets: Whether to use offsets from .mat file
        mat_filepath: Path to .mat file with xOffset and yOffset
        
    Returns:
        DataFrame with aligned coordinates
    """
    logger.info("Aligning gaze data to screen center")
    # Make a copy of the dataframe
    df_aligned = df.copy()
    
    # Only use fixation data for alignment
    df_fixations = df_aligned[df_aligned['Fixation'] == 1].copy()

    # Calculate the centroid of the gaze data
    centroid = calculate_centroid(df_fixations, threshold)

    # Calculate the difference between the centroid and the center
    diff = [screen_center[0] - centroid[0], screen_center[1] - centroid[1]]

    # Align the gaze data to the center
    df_aligned['X'] = df_aligned['X'] + diff[0]
    df_aligned['Y'] = df_aligned['Y'] + diff[1]

    return df_aligned