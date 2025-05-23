# Libraries
from typing import List, Dict
import datetime
import os
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
from data_preprocessing import merge_data

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
    symmetry = symmetry_ratio(left_stride_times, right_stride_times)
    
    output_dir = os.path.join(data_dir, 'gait_features')
    os.makedirs(output_dir, exist_ok=True) 
    
    
    # Save each metric into the gait_features directory
    pd.DataFrame({'left_stride_times': left_stride_times}).to_csv(os.path.join(output_dir, f'left_stride_{data_type}.csv'), index=False)
    pd.DataFrame({'right_stride_times': right_stride_times}).to_csv(os.path.join(output_dir, f'right_stride_{data_type}.csv'), index=False)
    pd.DataFrame(left_stance_swing).to_csv(os.path.join(output_dir, f'left_stance_swing_{data_type}.csv'), index=False)
    pd.DataFrame(right_stance_swing).to_csv(os.path.join(output_dir, f'right_stance_swing_{data_type}.csv'), index=False)
    pd.DataFrame({'asymmetry_index': asymmetry, 'symmetry_ratio': symmetry}).to_csv(os.path.join(output_dir, f'summary_gait_metrics_{data_type}.csv'), index=False)


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
    return [(l - r) / (l + r) if (l + r) != 0 else 0 for l, r in zip(left, right)]

def symmetry_ratio(left: list, right: list):
    '''
    Calculate the symmetry ratio between left and right stride times.
    Args:
        left (list): Left stride times.
        right (list): Right stride times.
    '''
    return [min(l, r) / max(l, r) if max(l, r) != 0 else 0 for l, r in zip(left, right)]

def summarize_metric(values: list[float]):
    '''
    Calculate the mean of a list of values, ignoring NaN values.
    Args:
        values (list[float]): List of values to summarize.
    '''
    return float(np.mean(values)) if len(values) > 0 else np.nan



# ------- Feature Extraction for Gait Detection  ----------------------------------------------------------------------

def generate_rolling_windows(patient_path: str, window_sec = 2, stride_sec = 1, fs = 100):
    '''
    Generate three possible datasets for real time gait asymmetry detection from Left and Right Shank raw data.
    Args:
        patient_path (str): Patient directory where the left and right shank data are stored.
        window_sec (int): Time window size of data.
        stride_sec (int): Time stride for each window of data.
        fs (int): Sampling frequency in Hz.
    '''
    # Merge the left and right accelerometer and gyroscope data into two files
    merge_data(patient_path, os.path.join(patient_path, 'LeftShank-Accelerometer.csv'), os.path.join(patient_path, 'RightShank-Accelerometer.csv'), 'accelerometer')
    merge_data(patient_path, os.path.join(patient_path, 'LeftShank-Gyroscope.csv'), os.path.join(patient_path, 'RightShank-Gyroscope.csv'), 'gyroscope')
    
    gyroscope_df     = pd.read_csv(os.path.join(patient_path, 'gyroscope.csv')).dropna()
    accelerometer_df = pd.read_csv(os.path.join(patient_path, 'accelerometer.csv')).dropna()
    
    # Define the window and stride size
    window_size = int(window_sec * fs)
    stride_size = int(stride_sec * fs)

    min_length = min(len(gyroscope_df), len(accelerometer_df))
    gyroscope_df = gyroscope_df.iloc[:min_length].reset_index(drop=True)
    accelerometer_df = accelerometer_df.iloc[:min_length].reset_index(drop=True)

    time_domain_windows = []
    asymmetry_domain_windows = []
    windows = []
    
    # Add the patient_id to ensure that we know the source, when data is merged to single dataset
    # and class label (healthy/stroke) to try and find any interesting correlations
    patient_id = patient_path.split('/')[-1].lower()
    status = 0 if patient_path.__contains__('Healthy') else 1
    patient_id = patient_id + '_' + str(status)
    
    for start_idx in range(0, min_length - window_size + 1, stride_size):
        end_idx = start_idx + window_size
        window_id  = len(windows)
        start_time = gyroscope_df.loc[start_idx, 'timestamp (+0700)'],
        end_time   = gyroscope_df.loc[end_idx - 1, 'timestamp (+0700)']
        
        # Calculate the symmetry/asymmetry gait metrics including stride times, stance/swing times, asymmetry index, and symmetry ratio for the current window
        left_peaks  = signal.find_peaks(flatten_list(gyroscope_df.loc[start_idx:end_idx - 1, ['left-z-axis (deg/s)']].values.tolist()),  height=0.3, distance=80)
        right_peaks = signal.find_peaks(flatten_list(gyroscope_df.loc[start_idx:end_idx - 1, ['right-z-axis (deg/s)']].values.tolist()), height=0.3, distance=80)
        left_stride_times  = np.diff(left_peaks[0])
        right_stride_times = np.diff(right_peaks[0])
        asymmetry = asymmetry_index(left_stride_times, right_stride_times)
        symmetry  = symmetry_ratio(left_stride_times, right_stride_times)
        
        if len(left_peaks[0]) > 1 and len(right_peaks[0]) > 1:
            label_strict   = 1 if abs(asymmetry[0]) > 0.2  or symmetry[0] < 0.8 else 0
            label_moderate = 1 if abs(asymmetry[0]) > 0.15 or symmetry[0] < 0.85 else 0
            label_lenient  = 1 if abs(asymmetry[0]) > 0.1  or symmetry[0] < 0.9 else 0
        else:
            label_strict   = 2
            label_moderate = 2
            label_lenient  = 2


        # Create the time domain features per window 
        time_window = {
            'patient_id'            : patient_id, 
            'window_id'             : window_id,
            'start_time'            : start_time[0],
            'end_time'              : end_time,
            'gyro-right-z-axis-max' : gyroscope_df.loc[start_idx:end_idx - 1, ['right-z-axis (deg/s)']].max().values[0],
            'gyro-left-z-axis-max'  : gyroscope_df.loc[start_idx:end_idx - 1, ['left-z-axis (deg/s)']].max().values[0],
            'gyro-right-z-axis-min' : gyroscope_df.loc[start_idx:end_idx - 1, ['right-z-axis (deg/s)']].min().values[0],
            'gyro-left-z-axis-min'  : gyroscope_df.loc[start_idx:end_idx - 1, ['left-z-axis (deg/s)']].min().values[0],
            'accel-right-z-axis-max': accelerometer_df.loc[start_idx:end_idx - 1, ['right-z-axis (g)']].max().values[0],
            'accel-left-z-axis-max' : accelerometer_df.loc[start_idx:end_idx - 1, ['left-z-axis (g)']].max().values[0],
            'accel-right-z-axis-min': accelerometer_df.loc[start_idx:end_idx - 1, ['right-z-axis (g)']].min().values[0],
            'accel-left-z-axis-min' : accelerometer_df.loc[start_idx:end_idx - 1, ['left-z-axis (g)']].min().values[0],
            'label_strict'          : label_strict, 
            'label_moderate'        : label_moderate, 
            'label_lenient'         : label_lenient,
            'class_label'           : status
        }
        time_domain_windows.append(time_window)
        
        # Create the asymmetry features per window
        asymmetry_window = {
            'patient_id'                      : patient_id, 
            'window_id'                       : window_id,
            'start_time'                      : start_time[0],
            'end_time'                        : end_time,
            'gyro-asymmetry-stride-times'     : summarize_metric(asymmetry),
            'gyro-symmetry-ratio-stride-times': summarize_metric(symmetry),
            'label_strict'                    : label_strict, 
            'label_moderate'                  : label_moderate, 
            'label_lenient'                   : label_lenient,
            'class_label'                     : status
        }
        asymmetry_domain_windows.append(asymmetry_window)
        
        # Create the raw data/features per window
        window = {
            'patient_id'    : patient_id,
            'window_id'     : window_id,
            'start_time'    : start_time[0],
            'end_time'      : end_time,
            'gyro_left'     : gyroscope_df.loc[start_idx:end_idx - 1, ['left-x-axis (deg/s)', 'left-y-axis (deg/s)', 'left-z-axis (deg/s)']].values,
            'gyro_right'    : gyroscope_df.loc[start_idx:end_idx - 1, ['right-x-axis (deg/s)', 'right-y-axis (deg/s)', 'right-z-axis (deg/s)']].values,
            'accel_left'    : accelerometer_df.loc[start_idx:end_idx - 1, ['left-x-axis (g)', 'left-y-axis (g)', 'left-z-axis (g)']].values,
            'accel_right'   : accelerometer_df.loc[start_idx:end_idx - 1, ['right-x-axis (g)', 'right-y-axis (g)', 'right-z-axis (g)']].values,
            'label_strict'  : label_strict, 
            'label_moderate': label_moderate, 
            'label_lenient' : label_lenient,
            'class_label'   : status
        }
        windows.append(window)
    
     
    # Delete the gyroscope and accelerometer files we created
    os.remove(os.path.join(patient_path, 'gyroscope.csv'))
    os.remove(os.path.join(patient_path, 'accelerometer.csv'))
    
    # Save time-domain feature dataset, asymmetry metric + label dataset
    pd.DataFrame(time_domain_windows).to_csv(os.path.join(patient_path, 'detection_time_domain.csv'), index=False)    
    pd.DataFrame(asymmetry_domain_windows).to_csv(os.path.join(patient_path, 'detection_asymmetry.csv'), index=False)
    
    # Save raw windows as NumPy tensor
    raw_array = []
    for w in windows:
        raw_tensor = np.hstack([w['gyro_left'], w['gyro_right'], w['accel_left'], w['accel_right']])
        raw_array.append(raw_tensor)
    raw_array = np.stack(raw_array)  # shape: (#windows, 200, 12)
    

    np.savez_compressed(
        os.path.join(patient_path, 'detection_raw_window.npz'),
        X              = raw_array,
        label_lenient  = np.array([w['label_lenient'] for w in windows]),
        label_moderate = np.array([w['label_moderate'] for w in windows]),
        label_strict   = np.array([w['label_strict'] for w in windows]),
        class_label    = np.array([w['class_label'] for w in windows]),
        patient_id     = np.array([w['patient_id'] for w in windows]),
        window_id      = np.array([w['window_id'] for w in windows])
    )
    
    print(f'Generated and saved all 3 datasets to: {patient_path}')

    
def flatten_list(gyro_data_list: list[list[float]]):
    '''
    Function to flatten the list of gyroscope data to get the peaks later
    Args:
        gyro_data_list (list[list[float]]): List of lists of float values 
    '''
    flattened_list = []
    for sublist in gyro_data_list:
        for item in sublist:
            flattened_list.append(item)
            
    return flattened_list

