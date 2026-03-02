"""
Gaze trajectory and spatial visualization.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import logging

logger = logging.getLogger(__name__)


def plot_gaze_trajectory(
    df: pd.DataFrame,
    trial: int = None,
    show_fixation_points: bool = True,
) -> Figure:
    """
    Plot gaze trajectory over time.
    
    Args:
        df: DataFrame with X, Y coordinates
        trial: Specific trial to plot (if None, plot all)
        show_fixation_points: Whether to overlay fixation point positions
        
    Returns:
        Matplotlib Figure object
    """
    # TODO: Implement from plot_gaze_data.py
    raise NotImplementedError("Gaze trajectory plot not yet implemented")


def plot_gaze_heatmap(
    df: pd.DataFrame,
    trial: int = None,
) -> Figure:
    """
    Plot 2D heatmap of gaze density.
    
    Args:
        df: DataFrame with X, Y coordinates
        trial: Specific trial to plot
        
    Returns:
        Matplotlib Figure object
    """
    # TODO: Implement heatmap visualization
    raise NotImplementedError("Gaze heatmap not yet implemented")
