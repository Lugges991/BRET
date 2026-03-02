"""
File parsing utilities.
"""

import pandas as pd
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)


def parse_trial_metadata_from_asc(
    filepath: Path,
) -> pd.DataFrame:
    """
    Extract trial metadata from .asc MSG lines.
    
    Args:
        filepath: Path to .asc file
        
    Returns:
        DataFrame with trial metadata (trial_nr, type, image, orientation, etc.)
    """
    # TODO: Implement from generate_consolidated_trial_data.py parse_asc_trials()
    raise NotImplementedError("Parse trial metadata not yet implemented")


def extract_run_info_from_filename(
    filename: str,
) -> dict:
    """
    Extract subject, run, and condition info from filename.
    
    Args:
        filename: Filename to parse
        
    Returns:
        Dictionary with 'subject', 'run', 'condition' keys
    """
    # TODO: Implement filename parsing
    # Examples: s11r02r, sub-11_run04_report_perceptData_20-Aug-2025_11_10_03.csv
    raise NotImplementedError("Extract run info not yet implemented")
