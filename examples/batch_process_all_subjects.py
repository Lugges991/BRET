"""
Example: Batch process all subjects.

This example demonstrates processing multiple subjects:
1. Find all subjects
2. Process each subject's runs
3. Generate quality reports
"""

from pathlib import Path
from bret import PreprocessingPipeline, load_config
from bret.utils.logging_setup import setup_logging


def main():
    # Setup logging
    setup_logging(level="INFO")
    
    # Load configuration
    config = load_config()
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(config)
    
    # Find all subject directories
    data_dir = Path("data")
    subject_dirs = sorted(data_dir.glob("sub-*"))
    
    print(f"Found {len(subject_dirs)} subjects")
    
    for subject_dir in subject_dirs:
        print(f"\nProcessing {subject_dir.name}...")
        
        try:
            # Process all runs for this subject
            results = pipeline.process_subject(
                subject_dir=subject_dir,
                output_dir=subject_dir / "preprocessed",
            )
            
            print(f"Successfully processed {len(results)} runs")
            
        except Exception as e:
            print(f"Error processing {subject_dir.name}: {e}")
            continue


if __name__ == "__main__":
    main()
