"""
Data validation utilities.
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def validate_file_exists(filepath: Path) -> bool:
    """
    Check if file exists and is readable.
    
    Args:
        filepath: Path to check
        
    Returns:
        True if file exists and is readable
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return True


def validate_data_schema(
    df: pd.DataFrame,
    required_columns: list,
    data_type: str = "preprocessed",
) -> bool:
    """
    Validate that DataFrame has required columns and structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        data_type: Type of data for error messages
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {data_type} data: {missing}")
    return True
