"""
Main preprocessing pipeline orchestration.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import logging
import re
from scipy.io import loadmat

from bret.preprocessing.parser import parse_asc_file
from bret.preprocessing.cleaning import interpolate_blinks
from bret.preprocessing.filtering import apply_butterworth_filter
from bret.preprocessing.alignment import align_to_center
from bret.preprocessing.validation import exclude_low_quality_trials, extract_trial_data
from bret.features.spatial import calculate_distance_to_fixation
from bret.utils import calculate_degrees_per_pixel

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for EyeLink data.
    
    Chains together parsing, cleaning, filtering, alignment, and validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config
        logger.info("Initialized preprocessing pipeline")
    
    def process_file(
        self,
        asc_file: Path,
        output_file: Path = None,
    ) -> pd.DataFrame:
        """
        Process a single .asc file through the complete pipeline.
        
        Args:
            asc_file: Path to input .asc file
            output_file: Path to save output CSV (optional)
            align: Whether to align gaze to screen center
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Processing {asc_file}")
        base_dir = asc_file.parent
        run_number = int(re.search(r"r(\d+)(?:nr|r)$", asc_file.stem).group(1))
        parameter_files = base_dir.glob("*stimuli.mat")
        # get all .mat files that contain "run"
        parameter_files = [f for f in parameter_files if "run" in f.stem]
        # get the mat file with highest run number that is less than or equal to the run number of the asc file
        m = re.search(r"r(\d+)(nr|r)$", asc_file.stem)
        run_number = int(m.group(1))
        run_type = "no-report" if m.group(2) == "nr" else "report"

        mat_file = None
        for f in parameter_files:
            f_nr = re.search(r"run(\d+)", f.stem)
            if f_nr:
                f_nr = int(f_nr.group(1))
                if f_nr<= run_number:
                    mat_file = f
                else:
                    break
        
        if mat_file is None:
            raise FileNotFoundError(
                f"No .mat parameter file found for {asc_file.name} "
                f"(run {run_number:02d}). Ensure a '*stimuli.mat' file exists."
            )

        mat_file = loadmat(mat_file)

        
        x_offset = mat_file["stimuli"]["xOffset"][0][0][0][0]
        y_offset = mat_file["stimuli"]["yOffset"][0][0][0][0]
        fixpoint_positions = self._extract_jumpFix_positions(mat_file)
        deg_per_pixel = calculate_degrees_per_pixel(
            screen_width_mm=self.config["width_mm"],
            screen_height_mm=self.config["height_mm"],
            screen_distance_mm=self.config["distance_mm"],
            resolution=self.config["screen_resolution"],
        )
        
        # 1. Parse .asc file
        df_parsed, metadata = parse_asc_file(asc_file)

        # throw out samples not in trials and get proper timestamps



        # 2. Interpolate blinks
        df_interpolated = interpolate_blinks(df_parsed, 
                                             self.config["diameter_threshold"],
                                             self.config["coalesce"],
                                             self.config["blink_padding"],
                                             self.config["saccade_padding"],
                                             metadata["sampling_rate"],
                                             self.config["screen_resolution"],
                                             x_offset,
                                             y_offset,
                                             )

        # 3. Apply Butterworth filter
        df_filtered = apply_butterworth_filter(df_interpolated,
                                                self.config["butter_order"],
                                                self.config["cutoff_frequency"],
                                                metadata["sampling_rate"],
                                                )
        threshold = np.sqrt(2 * self.config["fixation_distance"]**2)*2 # Twice the distance of the diagonal fixation spots to the center in pixels
        # 4. Align to center (if requested)
        df_aligned = align_to_center(df_filtered,
                                     threshold,
                                     (self.config["screen_resolution"][0] / 2, self.config["screen_resolution"][1] / 2),
                                     )

        # 5. Exclude low-quality trials
        df_out = exclude_low_quality_trials(df_aligned, self.config["missing_data_threshold"])
        df_out= extract_trial_data(df_out)

        # 6. Calculate spatial features
        df_out = calculate_distance_to_fixation(df_out, fixation_spots=fixpoint_positions, center=(self.config["screen_resolution"][0] / 2, self.config["screen_resolution"][1] / 2), deg_per_pix_width=deg_per_pixel)


        # 6. Save to output file (if specified)
        if output_file is not None:
            df_out.to_csv(output_file, index=False)
            logger.info(f"Saved preprocessed data to {output_file}")
    
    def process_subject(
        self,
        subject_dir: Path,
        output_dir: Path = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Process all .asc files for a subject.
        
        Args:
            subject_dir: Directory containing subject's .asc files
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary mapping run names to DataFrames
        """

        # get all subject .asc files
        asc_files = sorted(list(subject_dir.glob("*.asc")))
        # generate output names
        # out_names = [output_dir.joinpath(Path(f"{subject_dir.stem}_run-{re.search(r'r(\d+)(?:nr|r)$', f.stem).group(1)}_preprocessed.csv")) for f in asc_files]
        out_names = [output_dir.joinpath(str(a.stem) + "_preprocessed.csv") for a in asc_files]
        # call process_file for each
        for asc_file, output_file in zip(asc_files, out_names):
            self.process_file(asc_file, output_file)

    
    def _extract_jumpFix_positions(self, mat):
        """Extract 8x2 jump-fix positions from a raw scipy.io.loadmat dict."""
        stimuli = mat["stimuli"][0, 0]
        jumpFix = stimuli["jumpFix"][0, 0]

        raw = np.asarray(jumpFix["positions"]).squeeze()

        if raw.dtype != object:
            raise ValueError("positions is already numeric; likely indexed too deep.")

        if raw.shape == (8,):
            pos_list = [np.asarray(raw[i]).squeeze() for i in range(8)]
        elif raw.shape == (1, 8):
            pos_list = [np.asarray(raw[0, i]).squeeze() for i in range(8)]
        else:
            flat = raw.ravel()
            pos_list = [np.asarray(flat[i]).squeeze() for i in range(8)]

        pos_array = np.vstack(pos_list).astype(int)
        return pos_array