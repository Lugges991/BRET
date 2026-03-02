"""
Example: Custom reconstruction workflow using API.

This example demonstrates using the reconstruction API programmatically:
1. Load preprocessed data
2. Calculate features
3. Reconstruct percepts
4. Apply smoothing
5. Evaluate results
"""

from pathlib import Path
import pandas as pd
from bret import load_config
from bret.io.loaders import load_subject_preprocessed_data, load_percept_reports
from bret.reconstruction.euclidean import EuclideanReconstructor
from bret.reconstruction.smoothing import TemporalSmoother
from bret.reconstruction.evaluators import ReconstructionEvaluator
from bret.utils.logging_setup import setup_logging


def main():
    # Setup
    setup_logging(level="WARNING")
    config = load_config()
    
    # Load data
    subject_dir = Path("data/sub-11")
    breakpoint()
    
    dfs = load_subject_preprocessed_data(subject_dir)
    # reports = load_percept_reports(subject_dir, run_type="report")

    
    # Calculate features
    
    # Reconstruct percepts
    reconstructor = EuclideanReconstructor()
    processed_dfs = []
    for df in dfs:

        df = reconstructor.infer_percept_from_closest_fixpoint(df)
        df = reconstructor.infer_percept_with_mixed(df)
        processed_dfs.append(df)


    
    # Apply smoothing
    # smoother = TemporalSmoother()
    # optimal_params = smoother.grid_search_optimal_window(df)
    # df = smoother.apply_median_filter(df, window_size=optimal_params['window_size'])
    
    # # Evaluate
    # evaluator = ReconstructionEvaluator()
    # metrics = evaluator.evaluate(
    #     y_true=df['Reported Percept'],
    #     y_pred=df['InferredPerceptMixed'],
    # )
    
    # print("\nReconstruction Results:")
    # print(f"F1 Score (macro): {metrics['f1_macro']:.3f}")
    # print(f"Accuracy: {metrics['accuracy']:.3f}")
    # print(f"MCC: {metrics['mcc']:.3f}")


if __name__ == "__main__":
    main()
