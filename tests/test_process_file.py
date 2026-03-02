from bret.preprocessing.pipeline import PreprocessingPipeline
from pathlib import Path
import yaml

config = Path("bret/config/default_config.yaml")

# load yaml config
with open(config, "r") as f:
    config_dict = yaml.safe_load(f)


pipeline = PreprocessingPipeline(config_dict)

file_path = Path("data/sub-11/s11r02r.asc")
output_path = Path("testeroo.csv")

import time
start_time = time.time()
pipeline.process_file(file_path, output_file=output_path)
end_time = time.time()
print(f"Processing time: {end_time - start_time:.2f} seconds")