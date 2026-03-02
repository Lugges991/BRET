"""
Configuration loading and validation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (if None, use default)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Use default config from package
        package_dir = Path(__file__).parent.parent
        config_path = package_dir / "config" / "default_config.yaml"
    
    logger.info(f"Loading config from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # TODO: Add validation of config structure
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = [
        "preprocessing",
        "features",
        "reconstruction",
        "subjects",
        "paths",
    ]
    
    # TODO: Implement thorough validation
    raise NotImplementedError("Config validation not yet implemented")
