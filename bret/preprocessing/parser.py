"""
Parser for EyeLink .asc files.

Extracts sample data, events (fixations, saccades, blinks), and trial metadata
from raw EyeLink ASCII exports.
"""
import csv
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def parse_asc_file(filepath: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse an EyeLink .asc file into structured data.
    
    Args:
        filepath: Path to the .asc file
        
    Returns:
        Tuple of (samples_df, metadata_dict)
        - samples_df: DataFrame with columns [Timestamp, X, Y, Diameter, Fixation, Saccade, Blink, Trial, Type, Image1, Fixpoint1, Fixpoint2, Percept]
        - metadata_dict: Dict with 'gaze_coords', 'sampling_rate', 'n_saccades', 'n_fixations', 'n_blinks'
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    logger.info(f"Parsing {filepath}")
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Pre-compile regex patterns
    trial_pattern = re.compile(r'Trial #(\d+)')
    type_pattern = re.compile(r'type = (\w+)')
    image1_pattern = re.compile(r'image1 = (\w+)')
    fixpoint_pattern = re.compile(r'position: (\d+)')
    percept_pattern = re.compile(r'percept: (\w+)')
    
    # Initialize metadata
    gaze_coords = None
    sampling_rate = None
    
    # Initialize data storage (lists are faster than repeated DataFrame operations)
    samples_data = []
    fixation_intervals = []
    saccade_intervals = []
    blink_intervals = []
    
    # Current trial state
    trial_state = {
        'trial': None,
        'type': None,
        'image1': None,
        'fixpoint1': None,
        'fixpoint2': None,
        'percept': None
    }
    
    # Read file - much faster than csv.reader for this use case
    with open(filepath, 'r') as f:
        started = False
        
        for line in f:
            line = line.rstrip('\n')
            
            # Skip empty lines
            if not line:
                continue
            
            # Parse header before SYNCTIME
            if not started:
                if 'GAZE_COORDS' in line:
                    parts = line.split()
                    gaze_coords = [float(parts[i]) for i in range(3, min(6, len(parts)))]
                elif 'RATE' in line and 'SAMPLES' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'RATE' and i + 1 < len(parts):
                            sampling_rate = float(parts[i + 1])
                            break
                elif 'SYNCTIME' in line:
                    started = True
                continue
            
            # Split line into parts (tab-delimited)
            parts = line.split('\t')
            
            # Handle events (faster than multiple 'any' calls)
            first_part = parts[0]
            
            if first_part.startswith('EFIX'):
                start_tokens = first_part.split()
                if len(start_tokens) >= 3:
                    start_time = start_tokens[2]
                    end_time = parts[1]
                if len(parts) >= 3:
                    # Extract start time from first part (after 'EFIX L')
                    start_tokens = first_part.split()
                    if len(start_tokens) >= 3:  # ['EFIX', 'L', 'START_TIME']
                        start_time = float(start_tokens[2]) / 1000
                        end_time = float(parts[1]) / 1000
                        fixation_intervals.append((start_time, end_time))
                continue
                
            elif first_part.startswith('ESACC'):
                # ESACC L START END DURATION ...
                start_tokens = first_part.split()
                if len(start_tokens) >= 3:
                    start_time = float(start_tokens[2]) / 1000
                    end_time = float(parts[1]) / 1000
                    saccade_intervals.append((start_time,end_time))
                continue
                
            elif first_part.startswith('EBLINK'):
                message = " ".join(parts).split()
                blink_intervals.append((float(message[2]) / 1000, float(message[3]) / 1000))
                # EBLINK L START END DURATION
                #tokens = first_part.split(" L")
                #blink_intervals.append((float(tokens[1]) /1000, float(parts[-2])))
                
            elif first_part.startswith(('SFIX', 'SSACC', 'SBLINK')):
                # Skip start events
                continue
            
            # Handle messages
            elif 'MSG' in first_part:
                message = '\t'.join(parts[1:]) if len(parts) > 1 else ''
                
                if 'Trial #' in message:
                    match = trial_pattern.search(message)
                    trial_state['trial'] = int(match.group(1)) if match else None
                    
                    match = type_pattern.search(message)
                    trial_state['type'] = match.group(1) if match else None
                    
                    match = image1_pattern.search(message)
                    trial_state['image1'] = match.group(1) if match else None
                    trial_state['percept'] = trial_state['image1']  # Initial percept
                    
                elif 'End of trial' in message:
                    trial_state = {k: None for k in trial_state}
                    
                elif 'Fix point 1' in message:
                    match = fixpoint_pattern.search(message)
                    trial_state['fixpoint1'] = match.group(1) if match else None
                    
                elif 'Fix point 2' in message:
                    match = fixpoint_pattern.search(message)
                    trial_state['fixpoint2'] = match.group(1) if match else None
                    
                elif 'Current percept:' in message or 'percept:' in message:
                    match = percept_pattern.search(message)
                    trial_state['percept'] = match.group(1) if match else None
                
                continue
            
            elif 'END' in first_part:
                break
            
            # Parse sample data (timestamp, x, y, diameter)
            if len(parts) >= 4 and parts[0].strip().replace('.', '').isdigit():
                try:
                    timestamp = float(parts[0]) / 1000  # Convert ms to seconds
                    x = np.nan if parts[1].strip() == '.' else float(parts[1])
                    y = np.nan if parts[2].strip() == '.' else float(parts[2])
                    diameter = float(parts[3])
                    
                    # Append row with current trial state
                    samples_data.append([
                        timestamp, x, y, diameter,
                        trial_state['trial'],
                        trial_state['type'],
                        trial_state['image1'],
                        trial_state['fixpoint1'],
                        trial_state['fixpoint2'],
                        trial_state['percept']
                    ])
                except (ValueError, IndexError):
                    # Skip malformed sample lines
                    print("skipped")
                    continue

    # Create DataFrame from samples (much faster than iterative appends)
    df = pd.DataFrame(samples_data, columns=[
        'Timestamp', 'X', 'Y', 'Diameter', 'Trial', 'Type', 'Image1', 
        'Fixpoint1', 'Fixpoint2', 'Percept'
    ])
    
    logger.info(f"Parsed {len(df)} samples from {filepath.name}")
    
    # Create event indicator columns using vectorized operations
    df['Fixation'] = 0
    df['Saccade'] = 0
    df['Blink'] = 0
    
    # Mark events using interval matching (vectorized approach)
    timestamps = df['Timestamp'].values

    
    for start, end in fixation_intervals:
        mask = (timestamps >= start) & (timestamps <= end)
        df.loc[mask, 'Fixation'] = 1
        
    for start, end in saccade_intervals:
        mask = (timestamps >= start) & (timestamps <= end)
        df.loc[mask, 'Saccade'] = 1

        
    for start, end in blink_intervals:
        mask = (timestamps >= start) & (timestamps <= end)
        df.loc[mask, 'Blink'] = 1
    
    # Build metadata dict
    metadata = {
        'gaze_coords': gaze_coords,
        'sampling_rate': sampling_rate,
        'n_fixations': len(fixation_intervals),
        'n_saccades': len(saccade_intervals),
        'n_blinks': len(blink_intervals)
    }
    
    logger.info(f"Detected {metadata['n_fixations']} fixations, {metadata['n_saccades']} saccades, {metadata['n_blinks']} blinks")
    
    return df, metadata


def read_eyelink_data(path):
    """ 
    Read in the file and store the data in a dataframe. The file has the following structure:
    * header indicated with '**' at the beginning of each line
    * messages containing information about callibration/validation etc. indicated with 'MSG' at the beginning of each line
    
    This is followed by:
    START	10350638 	LEFT	SAMPLES	EVENT
    PRESCALER	1
    VPRESCALER	1
    PUPIL	AREA
    EVENTS	GAZE	LEFT	RATE	 1000.00	TRACKING	CR	FILTER	2
    SAMPLES	GAZE	LEFT	RATE	 1000.00	TRACKING	CR	FILTER	2
    
    This is followed by the actual data containing the following data types:
    * Samples: [TIMESTAMP]\t [X-Coords]\t [Y-Coords]\t [Diameter]
    * Messages: MSG [TIMESTAMP]\t [MESSAGE], e.g.
        - Trial #1: type = rivalry, report = 1, image1 = face, angle1 = 0°
        - Fix point 1 position: 3
        - Fix point 2 position: 6
        - Current percept: face
    * Events: 
        - SFIX (Start Fixation): SFIX [EYE (L/R)]\t [START TIME]\t 
        - EFIX (End Fixation): EFIX [EYE (L/R)]\t [START TIME]\t [END TIME]\t [DURATION]\t [AVG X]\t [AVG Y]\t [AVG PUPIL]\t
        - SSACC (Start Saccade): SSACC [EYE (L/R)]\t [START TIME]\t 
        - ESACC (End Saccade): ESACC [EYE (L/R)]\t [START TIME]\t [END TIME]\t [DURATION]\t [START X]\t [START Y]\t [END X]\t [END Y]\t [AMP]\t [PEAK VEL]\t
        - SBLINK (Start Blink): SBLINK [EYE (L/R)]\t [START TIME]\t 
        - EBLINK (End Blink): SBLINK [EYE (L/R)]\t [START TIME]\t [DURATION]

    Input: 
        path: str, path to the file
        
    Output:
        df: pd.DataFrame, dataframe containing the data
    """

    # Initialize dataframe
    df = pd.DataFrame()

    # Initialize lists to store the relevant data 
    timestamps = []
    x_coords = []
    y_coords = []
    diameters = []
    saccade_timestamps = []
    fixation_timestamps = []
    blink_timestamps = []
    trials = []
    types = []
    images1 = []
    fixpoint1_positions = []
    fixpoint2_positions = []
    percepts = []

    # Iterate over the file, extract the data, collect it into the lists and adjust the values if necessary
    try:
        with open(path) as f:
           file = csv.reader(f, delimiter='\t')
           start = False
           trial = None
           type = None
           image1 = None
           fixpoint1 = None
           fixpoint2 = None
           percept = None
           for i, row in enumerate(file):
                # Skip header (everything until message includes 'SYNCTIME')
                if not start: 
                    # Extract gaze coordinates
                    if any('GAZE_COORDS' in item for item in row):
                        gaze_coords = row[1].split(' ')[2:]
                        gaze_coords = [float(coord) for coord in gaze_coords]
                    # Extract sampling rate (number that follow 'RATE')
                    elif any('RATE' in item for item in row):
                        sampling_rate = float(row[4])
                    elif any('SYNCTIME' in item for item in row):
                        start = True
                    continue
                # Extract fixations, saccades, blinks, trials, messages and events
                if any('SFIX' in item for item in row): continue
                elif any('EFIX' in item for item in row):
                    fixation_timestamps.append([row[0].split(' ')[4], row[1]])
                    continue
                elif any('SSACC' in item for item in row): continue
                elif any('ESACC' in item for item in row):
                    saccade_timestamps.append([row[0].split(' ')[3], row[1]])
                    continue
                elif any('SBLINK' in item for item in row): continue
                elif any('EBLINK' in item for item in row):
                    blink_timestamps.append([row[0].split(' ')[2], row[1]])
                    continue
                # Extract trial information
                elif any('Trial' in item for item in row):
                    message = ''.join(row[1:]).strip() # Join string and remove 'MSG'
                    trial = re.search(r'Trial #(\d+)',  message).group(1) # Extract trial number
                    type = re.search(r'type = (\w+)', message).group(0).split(' = ')[1] # Extract trial type
                    image1 = re.search(r'image1 = (\w+)', message).group(0).split(' = ')[1] # Extract image1
                    percept = image1 # Set initial percept to image1
                    continue
                elif any('End of trial' in item for item in row):
                    trial = None
                    type = None
                    fixpoint1 = None
                    fixpoint2 = None
                    percept = None
                    continue
                elif any('Fix point 1' in item for item in row): 
                    message = ''.join(row[1:]).strip() # Join string and remove 'MSG'
                    fixpoint1 = re.search(r'position: (\d+)',  message).group(1) # Extract fix point position
                    continue
                elif any('Fix point 2' in item for item in row):
                    message = ''.join(row[1:]).strip() # Join string and remove 'MSG'
                    fixpoint2 = re.search(r'position: (\d+)',  message).group(1)
                    continue
                elif any('Current' in item for item in row): # extract word after ':' in 'Current percept: mixed'
                    message = ''.join(row[1:]).strip() # Join string and remove 'MSG'
                    percept = re.search(r'percept: (\w+)',  message).group(1)
                    continue
                elif any('MSG' in item for item in row):
                    continue
                # Stop at the end of the file
                elif any('END' in item for item in row):
                    break  
                # Extract timestamp and convert to float and divide by 1000 to convert from ms to s
                timestamp = row[0].strip()
                timestamp = float(timestamp)/1000
                timestamps.append(timestamp)
                # Extract x and y coordinates and convert to float. Set to NaN if data is missing
                x = row[1].strip()
                x = np.nan if x == '.' else float(x)
                x_coords.append(x)
                y = row[2].strip()
                y = np.nan if y == '.' else float(y)
                y_coords.append(y)  
                # Extract diameter and convert to float
                diameter = float(row[3].strip())  
                diameters.append(diameter)
                # Append trial information to lists
                trials.append(trial)
                types.append(type)
                images1.append(image1)
                fixpoint1_positions.append(fixpoint1)
                fixpoint2_positions.append(fixpoint2)
                percepts.append(percept)


        # For fixations, saccades and blinks create a list as long as the timestamps list with zeros
        fixations = [0] * len(timestamps)
        saccades = [0] * len(timestamps)
        blinks = [0] * len(timestamps)

        # Extract the start and end times of fixations, saccades and blinks and set the values in the respective lists to 1
        for i in range(len(fixation_timestamps)):
            start = float(fixation_timestamps[i][0])/1000
            end = float(fixation_timestamps[i][1])/1000
            for j in range(len(timestamps)):
                if timestamps[j] >= start and timestamps[j] <= end:
                    fixations[j] = 1
        for i in range(len(saccade_timestamps)):
            start = float(saccade_timestamps[i][0])/1000
            end = float(saccade_timestamps[i][1])/1000
            for j in range(len(timestamps)):
                if timestamps[j] >= start and timestamps[j] <= end:
                    saccades[j] = 1
        for i in range(len(blink_timestamps)):
            start = float(blink_timestamps[i][0])/1000
            end = float(blink_timestamps[i][1])/1000
            for j in range(len(timestamps)):
                if timestamps[j] >= start and timestamps[j] <= end:
                    blinks[j] = 1

        # Summarize data
        n_saccades = len(saccade_timestamps)
        n_fixations = len(fixation_timestamps)
        n_blinks = len(blink_timestamps)

        # Create a dataframe from the lists
        df = pd.DataFrame(list(zip(timestamps, x_coords, y_coords, diameters, fixations, saccades, blinks, trials, types, images1, fixpoint1_positions, fixpoint2_positions, percepts)), 
                   columns =['Timestamp', 'X', 'Y', 'Diameter', 'Fixation', 'Saccade', 'Blink', 'Trial', 'Type', 'Image1', 'Fixpoint1', 'Fixpoint2', 'Percept'])
        
        return df, gaze_coords, sampling_rate, n_saccades, n_fixations, n_blinks
    except Exception as e:
        raise Warning(f'Could not read ' + str(path) + ' properly! Error: {e}')
    