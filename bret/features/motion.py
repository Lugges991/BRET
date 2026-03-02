"""
Motion feature calculation.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_velocity(
    df: pd.DataFrame,
    dt: float = 0.004,
) -> pd.DataFrame:
    """
    Compute gaze velocity from position differences.
    
    Args:
        df: DataFrame with X and Y columns
        dt: Time step between samples (seconds)
        
    Returns:
        DataFrame with 'velocity' column
    """
    # TODO: Implement from mk_features.py compute_motion_features_fixed_dt()
    # Calculate Euclidean velocity from successive X/Y differences
    
    raise NotImplementedError("Velocity computation not yet implemented")


def compute_acceleration(
    df: pd.DataFrame,
    dt: float = 0.004,
) -> pd.DataFrame:
    """
    Compute gaze acceleration from velocity differences.
    
    Args:
        df: DataFrame with velocity column
        dt: Time step between samples (seconds)
        
    Returns:
        DataFrame with 'acceleration' column
    """
    # TODO: Implement acceleration calculation
    raise NotImplementedError("Acceleration computation not yet implemented")
