from pathlib import Path
from bret import PreprocessingPipeline, load_config
from bret.utils.logging_setup import setup_logging


def main():
    setup_logging(level="INFO")
    config = load_config()

    pipeline = PreprocessingPipeline(config)
    subject_dir = Path("data/sub-11")
    output_dir = subject_dir / "preprocessed"
    output_dir.mkdir(exist_ok=True)

    pipeline.process_subject(subject_dir=subject_dir, output_dir=output_dir)



if __name__ == "__main__":
    main()