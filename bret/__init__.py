"""
BRET: Binocular Rivalry Eye Tracking
=====================================

A Python package for preprocessing and analyzing eye tracking data from 
binocular rivalry experiments.

Main modules:
- preprocessing: Raw EyeLink data preprocessing pipeline
- reconstruction: Percept reconstruction from gaze patterns
- io: Data loading and saving utilities
- features: Feature engineering for gaze data
- utils: Utility functions and configuration
- visualization: Plotting and visualization tools
- cli: Command-line interface
"""

__version__ = "0.1.0"
__author__ = "Amelie BA"

from bret.utils.config_loader import load_config
from bret.preprocessing.pipeline import PreprocessingPipeline
from bret.reconstruction.euclidean import EuclideanReconstructor

__all__ = [
    "load_config",
    "PreprocessingPipeline",
    "EuclideanReconstructor",
]
