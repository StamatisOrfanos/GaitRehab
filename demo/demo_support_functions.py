# Libraries
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks



def time_domain_features(data: pd.DataFrame):
    '''
    Calculate the metrics for the gyroscope data.
    Args:
        data (pd.Dataframe): Gyroscope data dataframe including the columns 'right_min', 'left_min', 'right_max', and 'left_max
    '''
    model_input = []
    
    # Calculate the time domain metrics
    model_input.append(data['right-z-axis (deg/s)'].min())
    model_input.append(data['left-z-axis (deg/s)'].min())
    model_input.append(data['right-z-axis (deg/s)'].max())
    model_input.append(data['left-z-axis (deg/s)'].max())

    return model_input

def calculate_gait_cycles():
    '''
    
    '''
    
    
# --------------------------------------------------------------------------------------------------------------------------------------------

def extract_features_per_gait_cycle(data):
    '''
    Extract features from gyroscope data for each gait cycle in a patient's folder.
    Args:
        patient_folder (str): Path to the patient's folder containing gyroscope data.
    '''
    result = []
    
    data['timestamp (+0700)'] = pd.to_datetime(data['timestamp (+0700)'])

    left_z        = data['left-z-axis (deg/s)'].values
    left_peaks, _ = find_peaks(left_z, height=0.5, distance=80)

    for i in range(len(left_peaks) - 1):
        
        start = left_peaks[i]
        end = left_peaks[i + 1]

        window = data.iloc[start:end]
        left_z_axis  = window['left-z-axis (deg/s)'].values
        right_z_axis = window['right-z-axis (deg/s)'].values
        time         = window['timestamp (+0700)']
        
        left_phases = detect_stance_swing(left_z_axis, time)
        right_phases = detect_stance_swing(right_z_axis, time)

        f = {
            'right_stance_time': np.mean([p['stance_time'] for p in right_phases]) if right_phases else np.nan,
            'left_stance_time' : np.mean([p['stance_time'] for p in left_phases])  if left_phases else np.nan,
            'right_swing_time' : np.mean([p['swing_time'] for p in right_phases])  if right_phases else np.nan,
            'left_swing_time'  : np.mean([p['swing_time'] for p in left_phases])   if left_phases else np.nan,
            'right_motion_score': motion_score(right_z_axis)
        }

        result.append(f)


def detect_stance_swing(z_filtered, time):
    '''
    Vectorized stance and swing phase detection from filtered z-axis gyro signal.
    Args:
        z_filtered (np.array): Filtered z-axis gyroscope data.
        time (pd.Series): Corresponding time values (datetime).
    '''
    time = pd.Series(time).reset_index(drop=True)
    zero_crossings = np.where(np.diff(np.sign(z_filtered)))[0]
    if len(zero_crossings) < 2:
        return []

    start_idxs = zero_crossings[:-1]
    end_idxs   = zero_crossings[1:]

    valid_pairs  = [(s, e) for s, e in zip(start_idxs, end_idxs) if e - s > 1]
    stance_times = []
    swing_times  = []

    for start, end in valid_pairs:
        min_idx = np.argmin(z_filtered[start:end]) + start
        
        if min_idx >= len(time) or start >= len(time) or end >= len(time):
            continue
        
        stance_time = (time.iloc[min_idx] - time.iloc[start]).total_seconds()
        swing_time  = (time.iloc[end] - time.iloc[min_idx]).total_seconds()
        stance_times.append(stance_time)
        swing_times.append(swing_time)

    return [{'stance_time': s, 'swing_time': w} for s, w in zip(stance_times, swing_times)]


def motion_score(z_signal: np.ndarray):
    '''
    Calculate the motion score for a z-axis gyroscope signal.
    Args:
        z_signal (np.ndarray): Z-axis gyroscope signal.
    '''
    return np.max(np.abs(z_signal)) - np.min(np.abs(z_signal))