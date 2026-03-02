"""
Example: Preprocess a single .asc file.

This example demonstrates the basic preprocessing workflow:
1. Load configuration
2. Create preprocessing pipeline
3. Process a single file
4. Save output
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
    
    # Define paths
    data_dir = Path("data/sub-11")
    asc_file = data_dir / "s11r02r.asc"
    output_file = data_dir / "preprocessed" / "sub-11_run-02_report_preprocessed.csv"
    
    # Process file
    df = pipeline.process_file(
        asc_file=asc_file,
        output_file=output_file,
        align=True,
    )
    
    print(f"Preprocessed {len(df)} samples")
    print(f"Output saved to {output_file}")


if __name__ == "__main__":
    main()
