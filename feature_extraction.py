# Libraries
from typing import List
import datetime
import os
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
from data_preprocessing import merge_data
from scipy.stats import skew, kurtosis, iqr
from scipy import signal
from scipy.stats import iqr, skew, kurtosis


# ------- Feature Extraction Functions for Classification ----------------------------------------------------------------------

def time_domain_features(data_dir: str, data_type: str):
    '''
    Calculate the metrics for the gyroscope/accelerometer data.
    Mean, Standard Deviation, Maximum, Minimum, Root Mean Square, Median Absolute Deviation, Range,
    Interquartile Range, Skewness & Kurtosis, Zero-crossing rate, Peak count / amplitude
    Args:
        data_dir (str): Directory where the merged gyroscope/accelerometer data is saved.
        data_type (str): Type of data to be merged (accelerometer or gyroscope).
    '''
    # Check if the files exist
    if not os.path.exists(data_dir + '{}.csv'.format(data_type)):
        raise FileNotFoundError(f'{data_type}.csv file not found.')
    
    # Read the merged gyroscope/accelerometer data
    data = pd.read_csv(data_dir + '{}.csv'.format(data_type))
    data.dropna(inplace=True)
    
    # Calculate the time domain metrics
    metrics = {}
    measurement_unit = 'deg/s' if data_type == 'gyroscope' else 's'
    
    for side in ['left', 'right']:
        z_axis = data[f'{side}-z-axis (deg/s)'] if data_type == 'gyroscope' else data[f'{side}-z-axis (s)'] 
        metrics[f'{side}-z-axis-({measurement_unit})-mean']  = z_axis.mean()
        metrics[f'{side}-z-axis-({measurement_unit})-std']   = z_axis.std()
        metrics[f'{side}-z-axis-({measurement_unit})-max']   = z_axis.max()
        metrics[f'{side}-z-axis-({measurement_unit})-min']   = z_axis.min()        
        metrics[f'{side}-z-axis-({measurement_unit})-rms']   = np.sqrt(np.mean(z_axis ** 2))
        metrics[f'{side}-z-axis-({measurement_unit})-mad']   = np.median(np.abs(z_axis - np.median(z_axis)))
        metrics[f'{side}-z-axis-({measurement_unit})-range'] = metrics[f'{side}-z-axis-({measurement_unit})-max'] - metrics[f'{side}-z-axis-({measurement_unit})-min']
        metrics[f'{side}-z-axis-({measurement_unit})-iqr']   = np.percentile(z_axis, 75) - np.percentile(z_axis, 25)
        metrics[f'{side}-z-axis-({measurement_unit})-skew']  = ((z_axis - z_axis.mean())**3).mean() / (z_axis.std()**3)
        metrics[f'{side}-z-axis-({measurement_unit})-kurt']  = ((z_axis - z_axis.mean())**4).mean() / (z_axis.std()**4)
        metrics[f'{side}-z-axis-({measurement_unit})-zcr']   = ((z_axis[:-1] * z_axis[1:]) < 0).sum()
        metrics[f'{side}-z-axis-({measurement_unit})-pkcnt'] = ((z_axis[:-1] * z_axis[1:]) < 0).sum()
        metrics[f'{side}-z-axis-({measurement_unit})-pkamp'] = z_axis.max() - z_axis.min()
    
    # Save the metrics to a CSV file
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(os.path.join(data_dir + f'time_domain_metrics_{data_type}.csv'), index=False)
    

def frequency_domain_features(data_dir: str, data_type: str, fs=100, window_duration_sec=2):
    '''
    Calculate the frequency domain features for the gyroscope data.
    Dominant frequency, Spectral entropy, Gait band energy
    Args:
        data_dir (str): Path to the gyroscope data CSV file.
        data_type (str): Type of data to be merged (accelerometer or gyroscope).
        fs (int): Sampling frequency in Hz.
        window_duration_sec (int): Duration of the window in seconds.
    '''
    # Check if the files exist
    if not os.path.exists(data_dir + '{}.csv'.format(data_type)):
        raise FileNotFoundError(f'{data_type}.csv file not found.')
    
    # Read data
    data = pd.read_csv(data_dir + '{}.csv'.format(data_type))
    data['timestamp (+0700)'] = pd.to_datetime(data['timestamp (+0700)'])
    data.dropna(inplace=True)

    start_time = data['timestamp (+0700)'].iloc[0]
    end_time   = data['timestamp (+0700)'].iloc[-1]
        
    window_features = []
    window_id = 0
    delta = pd.Timedelta(seconds=window_duration_sec)
    current_start = start_time

    # Loop through the data in windows of 2 seconds and get the important frequency domain features
    while current_start + delta <= end_time:
        current_end = current_start + delta
        window = data[(data['timestamp (+0700)'] >= current_start) & (data['timestamp (+0700)'] < current_end)]

        for side in ['left', 'right']:
            z_axis = f'{side}-z-axis (deg/s)' if data_type == 'gyroscope' else f'{side}-z-axis (s)'
            
            signal = window[z_axis].values
            if len(signal) < 2: continue

            fft_values  = fft(signal)
            frequencies = fftfreq(len(signal), d=1/fs)
            power_spectral_density = np.abs(fft_values)**2

            pos_frequencies = frequencies[:len(signal) // 2]
            pos_power_spectral_density   = power_spectral_density[:len(signal) // 2]

            dominant_freq = pos_frequencies[np.argmax(pos_power_spectral_density)]
            power_spectral_density_norm = pos_power_spectral_density / np.sum(pos_power_spectral_density)
            spectral_entropy = -np.sum(power_spectral_density_norm * np.log2(power_spectral_density_norm + 1e-10))
            gait_band_energy = np.sum(pos_power_spectral_density[(pos_frequencies >= 0.5) & (pos_frequencies <= 3)])

            window_features.append({
                'window_id': window_id,
                'side': side,
                'start_time': current_start,
                'end_time': current_end,
                'dominant_freq': dominant_freq,
                'spectral_entropy': spectral_entropy,
                'gait_band_energy': gait_band_energy,
                'samples': len(signal)
            })
        window_id += 1
        current_start += delta

    features_df = pd.DataFrame(window_features)
    features_df.to_csv(os.path.join(data_dir, f'windowed_frequency_features_{data_type}.csv'), index=False)

    
def gait_features(data_dir: str, data_type: str):
    '''
    Calculate the gait features for the gyroscope data.
    Stride times, Stance/swing times, Asymmetry index, Symmetry ratio
    Args:
        data_dir (str): Directory where the merged gyroscope data is saved.
        data_type (str): Type of data to be merged (accelerometer or gyroscope).
    '''
    # Check if the files exist
    if not os.path.exists(data_dir + '{}.csv'.format(data_type)):
        raise FileNotFoundError(f'{data_type}.csv file not found.')
    
    # Read the merged gyroscope/accelerometer data
    data = pd.read_csv(data_dir + '{}.csv'.format(data_type))
    data.dropna()
    
    # Calculate the gait metrics including stride times, stance/swing times, asymmetry index, and symmetry ratio
    left_peaks  = signal.find_peaks(data['left-z-axis (deg/s)'], height=0.5, distance=100)
    right_peaks = signal.find_peaks(data['right-z-axis (deg/s)'], height=0.5, distance=100)
    left_stride_times  = np.diff(left_peaks[0])
    right_stride_times = np.diff(right_peaks[0])
    left_stance_swing  = detect_stance_swing_fast(data['left-z-axis (deg/s)'], data['timestamp (+0700)'])
    right_stance_swing = detect_stance_swing_fast(data['right-z-axis (deg/s)'], data['timestamp (+0700)'])
    asymmetry = asymmetry_index(left_stride_times, right_stride_times)
    symmetry  = symmetry_ratio(left_stride_times, right_stride_times)  
    
    output_dir = os.path.join(data_dir, 'gait_features')
    os.makedirs(output_dir, exist_ok=True) 
    
    # Save each metric into the gait_features directory
    pd.DataFrame({'left_stride_times': left_stride_times}).to_csv(os.path.join(output_dir, f'left_stride_{data_type}.csv'), index=False)
    pd.DataFrame({'right_stride_times': right_stride_times}).to_csv(os.path.join(output_dir, f'right_stride_{data_type}.csv'), index=False)
    pd.DataFrame(left_stance_swing).to_csv(os.path.join(output_dir, f'left_stance_swing_{data_type}.csv'), index=False)
    pd.DataFrame(right_stance_swing).to_csv(os.path.join(output_dir, f'right_stance_swing_{data_type}.csv'), index=False)

    symmetry_ratios = []
    for l, r in zip(left_stride_times, right_stride_times):
        if max(l, r) != 0:
            symmetry_ratios.append(min(l, r) / max(l, r))
        else:
            symmetry_ratios.append(0)

    pd.DataFrame({'symmetry_ratio': symmetry_ratios}).to_csv(os.path.join(output_dir, f'summary_gait_metrics_{data_type}.csv'), index=False)

    asymmetry = asymmetry_index(left_stride_times, right_stride_times)
    symmetry = symmetry_ratio(left_stride_times, right_stride_times)
    pd.DataFrame({'asymmetry_index': [asymmetry], 'symmetry_ratio': [symmetry]}).to_csv(
        os.path.join(output_dir, f'summary_gait_metrics_overall_{data_type}.csv'), index=False
    )

def cross_limb_features(data_dir: str, data_type: str, fs=100):
    '''
    Calculate the cross limb features for the gyroscope data.
    Left and right stride durations, stride duration difference, stride duration symmetry ratio,
    Args:
        data_dir (str): Directory where the merged gyroscope data is saved.
        data_type (str): Type of data to be merged (accelerometer or gyroscope).
        fs (int): Sampling frequency in Hz.
    '''
    # Check if the files exist
    if not os.path.exists(data_dir + '{}.csv'.format(data_type)):
        raise FileNotFoundError(f'{data_type}.csv file not found.')
    
    # Read the merged gyroscope/accelerometer data
    data = pd.read_csv(data_dir + '{}.csv'.format(data_type))
    data.dropna(inplace=True)    
    
    data['left_z_filtered']  = butter_low_pass(data['left-z-axis (deg/s)'], fs=fs)
    data['right_z_filtered'] = butter_low_pass(data['right-z-axis (deg/s)'], fs=fs)
    
    left_peaks, _  = find_peaks(data['left_z_filtered'], distance=fs*0.5)
    right_peaks, _ = find_peaks(data['right_z_filtered'], distance=fs*0.5)
    
    features = []
    for i in range(min(len(left_peaks), len(right_peaks)) - 1):
        l_start, l_end = left_peaks[i], left_peaks[i+1]
        r_start, r_end = right_peaks[i], right_peaks[i+1]

        l_cycle = data.iloc[l_start:l_end]
        r_cycle = data.iloc[r_start:r_end]

        # Truncate to shortest cycle
        min_len = min(len(l_cycle), len(r_cycle))
        if min_len < 5: continue

        left_stride_duration  = data['left-elapsed (s)'].iloc[l_end]  - data['left-elapsed (s)'].iloc[l_start]
        right_stride_duration = data['right-elapsed (s)'].iloc[r_end] - data['right-elapsed (s)'].iloc[r_start]

        feature = {
            'left_stride_duration': left_stride_duration,
            'right_stride_duration': right_stride_duration,
            'stride_duration_diff': abs(left_stride_duration - right_stride_duration),
            'stride_duration_symmetry_ratio': min(left_stride_duration, right_stride_duration) / max(left_stride_duration, right_stride_duration),
            'left_peak': data['left_z_filtered'].iloc[l_start:l_end].max(),
            'right_peak': data['right_z_filtered'].iloc[r_start:r_end].max(),
            'peak_diff': abs(data['left_z_filtered'].iloc[l_start:l_end].max() - data['right_z_filtered'].iloc[r_start:r_end].max()),
            'z_corr': np.corrcoef(
                l_cycle['left_z_filtered'].values[:min_len],
                r_cycle['right_z_filtered'].values[:min_len]
            )[0,1]
        }
        features.append(feature)

    # Save the cross limb metrics to a CSV file
    cross_limb_features = pd.DataFrame(features)
    cross_limb_features.to_csv(os.path.join(data_dir, 'cross_limb_metrics.csv'), index=False)


def butter_low_pass(data: np.array, cutoff=6, fs=100, order=2):
    '''
    Apply a low-pass Butterworth filter to the data.
    Args:
        data (np.array): Input data to be filtered.
        cutoff (float): Cutoff frequency in Hz.
        fs (int): Sampling frequency in Hz.
        order (int): Order of the filter.
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def detect_stance_swing_fast(z_filtered: np.array, time: np.array):
    '''
    Vectorized stance and swing phase detection from filtered z-axis gyro signal.
    Args:
        z_filtered (np.array): Filtered z-axis gyroscope data.
        time (np.array or Series): Corresponding time values.
    '''
    time = pd.to_datetime(pd.Series(time))    
    zero_crossings = np.where(np.diff(np.sign(z_filtered)))[0]
    if len(zero_crossings) < 2: return []
    
    start_idxs = zero_crossings[:-1]
    end_idxs = zero_crossings[1:]
    
    valid_pairs = [(s, e) for s, e in zip(start_idxs, end_idxs) if e - s > 1]
    stance_times = []
    swing_times = []

    for start, end in valid_pairs:
        min_idx = np.argmin(z_filtered[start:end]) + start
        stance = (time.iloc[min_idx] - time.iloc[start]).total_seconds()
        swing  = (time.iloc[end] - time.iloc[min_idx]).total_seconds()
        stance_times.append(stance)
        swing_times.append(swing)

    return [{'stance_time': st, 'swing_time': sw} for st, sw in zip(stance_times, swing_times)]


def asymmetry_index(left: list, right: list):
    '''
    Calculate the asymmetry index between left and right stride times.
    Args:
        left (list): Left stride times.
        right (list): Right stride times.
    '''
    if len(left) == 0 or len(right) == 0: return np.nan
    return np.mean([(l - r) / (l + r) if (l + r) != 0 else 0 for l, r in zip(left, right)])

def symmetry_ratio(left: list, right: list):
    '''
    Calculate the symmetry ratio between left and right stride times.
    Args:
        left (list): Left stride times.
        right (list): Right stride times.
    '''
    if len(left) == 0 or len(right) == 0: return np.nan
    return np.mean([min(l, r) / max(l, r) if max(l, r) != 0 else 0 for l, r in zip(left, right)])


# ------- Feature Extraction for Gait Detection  ----------------------------------------------------------------------

def extract_features_per_gait_cycle(patient_folder: str, fs: int=100):
    '''
    Extract features from gyroscope data for each gait cycle in a patient's folder.
    Args:
        patient_folder (str): Path to the patient's folder containing gyroscope data.
    '''
    print(f'Processing {patient_folder}...')

    merge_data(patient_folder, os.path.join(patient_folder, 'LeftShank-Gyroscope.csv'), os.path.join(patient_folder, 'RightShank-Gyroscope.csv'), 'gyroscope')
    gyro_path = os.path.join(patient_folder, 'gyroscope.csv')
    
    if not os.path.exists(gyro_path):
        print(f'Skipping {patient_folder}, gyroscope.csv not found.')
        return

    # Load data, fix timestamp type and use a low-pass Butterworth filter 
    data = pd.read_csv(gyro_path).dropna()
    data['timestamp (+0700)']    = pd.to_datetime(data['timestamp (+0700)'])
    data['left-z-axis (deg/s)']  = butter_low_pass(data['left-z-axis (deg/s)'], fs=fs)
    data['right-z-axis (deg/s)'] = butter_low_pass(data['right-z-axis (deg/s)'], fs=fs)

    status     = '0' if 'Healthy' in patient_folder else '1'
    patient_id = f"{os.path.basename(patient_folder)}_{status}"
    results = []

    left_z        = data['left-z-axis (deg/s)'].values
    left_peaks, _ = find_peaks(left_z, height=0.5, distance=80)

    for i in range(len(left_peaks) - 1):
        
        start = left_peaks[i]
        end = left_peaks[i + 1]
        if end - start < 10: continue

        window = data.iloc[start:end]
        left_z_axis  = window['left-z-axis (deg/s)'].values
        right_z_axis = window['right-z-axis (deg/s)'].values
        time         = window['timestamp (+0700)']

        f = {
            'patient_id': patient_id,
            'window_id': i,
            'start_time': time.iloc[0],
            'end_time': time.iloc[-1],
            'left_z_mean': left_z_axis.mean(),  
            'left_z_std': left_z_axis.std(),    
            'left_z_max': left_z_axis.max(),    
            'left_z_min': left_z_axis.min(),    
            'left_motion_score': motion_score(left_z_axis),
            'right_z_mean': right_z_axis.mean(),
            'right_z_std': right_z_axis.std(),  
            'right_z_max': right_z_axis.max(),  
            'right_z_min': right_z_axis.min(),  
            'right_motion_score': motion_score(right_z_axis)
        }

        left_phases = detect_stance_swing(left_z_axis, time)
        right_phases = detect_stance_swing(right_z_axis, time)

        f['left_stance_time'] = np.mean([p['stance_time'] for p in left_phases])  if left_phases else np.nan
        f['left_swing_time']  = np.mean([p['swing_time'] for p in left_phases])   if left_phases else np.nan
        f['right_stance_time']= np.mean([p['stance_time'] for p in right_phases]) if right_phases else np.nan
        f['right_swing_time'] = np.mean([p['swing_time'] for p in right_phases])  if right_phases else np.nan

        left_stride  = (f['left_stance_time'] + f['left_swing_time']) if pd.notna(f['left_stance_time']) and pd.notna(f['left_swing_time']) else np.nan
        right_stride = (f['right_stance_time'] + f['right_swing_time']) if pd.notna(f['right_stance_time']) and pd.notna(f['right_swing_time']) else np.nan

        if pd.isna(left_stride) or pd.isna(right_stride):
            f['valid_gait_window'] = 0
            f['asymmetry_index'] = np.nan
            f['symmetry_ratio']  = np.nan
            f['label_high_confidence']     = -1
            f['label_moderate_confidence'] = -1
            f['label_low_confidence']      = -1
        else:
            f['valid_gait_window'] = 1
            f['asymmetry_index'] = single_asymmetry_index([left_stride], [right_stride])
            f['symmetry_ratio']  = single_symmetry_ratio([left_stride], [right_stride])
            f['label_high_confidence']     = 1 if abs(f['asymmetry_index']) > 0.2  or f['symmetry_ratio'] < 0.8 else 0
            f['label_moderate_confidence'] = 1 if abs(f['asymmetry_index']) > 0.15 or f['symmetry_ratio'] < 0.85 else 0
            f['label_low_confidence']      = 1 if abs(f['asymmetry_index']) > 0.1  or f['symmetry_ratio'] < 0.9 else 0

        results.append(f)

    result_df = pd.DataFrame(results)
    out_path = os.path.join(patient_folder, 'detection.csv')
    result_df.to_csv(out_path, index=False)
    print(f'Saved {len(result_df)} gait cycles to {out_path}')
    os.remove(os.path.join(patient_folder, 'gyroscope.csv'))

def motion_score(z_signal: np.ndarray):
    '''
    Calculate the motion score for a z-axis gyroscope signal.
    Args:
        z_signal (np.ndarray): Z-axis gyroscope signal.
    '''
    return np.max(np.abs(z_signal)) - np.min(np.abs(z_signal))


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

def single_asymmetry_index(left: list, right: list):
    '''
    Calculate the asymmetry index between left and right stride times.
    Args:
        left (list): Left stride times.
        right (list): Right stride times.
    '''
    if len(left) == 0 or len(right) == 0: return np.nan
    return np.mean([(l - r) / (l + r) if (l + r) != 0 else 0 for l, r in zip(left, right)])

def single_symmetry_ratio(left: list, right: list):
    '''
    Calculate the symmetry ratio between left and right stride times.
    Args:
        left (list): Left stride times.
        right (list): Right stride times.
    '''
    if len(left) == 0 or len(right) == 0: return np.nan
    return np.mean([min(l, r) / max(l, r) if max(l, r) != 0 else 0 for l, r in zip(left, right)])

def summarize_metric(x):
    '''
    Calculate the mean, standard deviation, min, max, skewness, kurtosis, and interquartile range of a list of values.
    Args:
        x (list or np.array): List or array of values to summarize.
    '''
    x = np.array(x)
    return [np.mean(x), np.std(x), np.min(x), np.max(x), skew(x), kurtosis(x), iqr(x)] if len(x) > 0 else [0]*7