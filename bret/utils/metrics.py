"""
Shared metric calculation functions.
"""

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
import logging

logger = logging.getLogger(__name__)


def compute_f1_score(
    y_true,
    y_pred,
    average: str = "macro",
):
    """
    Compute F1 score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('macro', 'weighted', 'micro')
        
    Returns:
        F1 score
    """
    return f1_score(y_true, y_pred, average=average)


def compute_mcc(y_true, y_pred):
    """
    Compute Matthews Correlation Coefficient.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        MCC score
    """
    return matthews_corrcoef(y_true, y_pred)
