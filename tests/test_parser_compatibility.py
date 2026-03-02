from bret.preprocessing.parser import parse_asc_file, read_eyelink_data
from pathlib import Path
import time

filepath = Path("data/sub-11/s11r02r.asc")

# time the two functions
start_time = time.time()
df_new, metadata = parse_asc_file(filepath)
end_time = time.time()
print(f"parse_asc_file took {end_time - start_time:.4f} seconds")
print(f"Metadata: {metadata}")

start_time = time.time()
df_old, gaze_coords, sampling_rate, n_saccades, n_fixations, n_blinks = read_eyelink_data(str(filepath))
end_time = time.time()
print(f"read_eyelink_data took {end_time - start_time:.4f} seconds")
print(f"Gaze coordinates shape: {gaze_coords}", f"Sampling rate: {sampling_rate}", f"Number of saccades: {n_saccades}", f"Number of fixations: {n_fixations}", f"Number of blinks: {n_blinks}")
