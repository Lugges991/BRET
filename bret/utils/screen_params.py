"""
Screen parameter calculations.
"""

import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_degrees_per_pixel(
    screen_width_mm: float,
    screen_height_mm: float,
    screen_distance_mm: float,
    resolution: Tuple[int, int],
) -> float:
    """
    Calculate degrees of visual angle per pixel.
    
    Args:
        screen_width_mm: Physical screen width in mm
        screen_height_mm: Physical screen height in mm
        screen_distance_mm: Viewing distance in mm
        resolution: Screen resolution (width, height) in pixels
        
    Returns:
        Degrees per pixel conversion factor
    """
    screen_width_px = float(resolution[0])
    screen_height_px = float(resolution[1])

    pix_width_mm = screen_width_mm / screen_width_px
    pix_height_mm = screen_height_mm / screen_height_px

    deg_per_pix_width = 2.0 * np.degrees(np.arctan((0.5 * pix_width_mm) / screen_distance_mm))
    deg_per_pix_height = 2.0 * np.degrees(np.arctan((0.5 * pix_height_mm) / screen_distance_mm))

    if not np.isclose(deg_per_pix_width, deg_per_pix_height):
        logger.debug(
            "deg_per_pix width/height differ: width=%s, height=%s",
            deg_per_pix_width,
            deg_per_pix_height,
        )

    return float(deg_per_pix_width)


def define_fixation_spot_positions(
    distance: float = 56,
    positions: str = "cardinal_diagonal",
) -> List[Tuple[float, float]]:
    """
    Define positions of fixation spots around center.
    
    Args:
        distance: Distance from center in pixels
        positions: Position configuration ('cardinal', 'diagonal', 'cardinal_diagonal')
        
    Returns:
        List of (x, y) coordinate tuples
    """
    # TODO: Implement from analysis.py get_data()
    # 8 positions: 4 cardinal (N, S, E, W) + 4 diagonal (NE, SE, SW, NW)
    
    raise NotImplementedError("Fixation positions not yet implemented")
