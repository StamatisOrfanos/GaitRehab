# Libraries
import os
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq


def merge_data(data_dir, left_shank_path, right_shank_path, merge_type):
    '''
    Merge gyroscope data from left and right shank.
    Args:
        data_dir (str): Directory where the merged gyroscope data will be saved.
        left_shank_path (str): Path to the left shank gyroscope CSV file.
        right_shank_path (str): Path to the right shank gyroscope CSV file.
        merge_type (str): Type of data to be merged (accelerometer or gyroscope).
    '''
    # Check if the files exist
    if not os.path.exists(left_shank_path):
        raise FileNotFoundError('LeftShank-Gyroscope.csv file not found.')
    if not os.path.exists(right_shank_path):
        raise FileNotFoundError('RightShank-Gyroscope.csv file not found.')

    # Read the CSV files
    left_shank_data  = pd.read_csv(left_shank_path)
    right_shank_data = pd.read_csv(right_shank_path)
        
    if merge_type == 'accelerometer':
        left_columns_names = {'elapsed (s)': 'left-elapsed (s)', 'x-axis (s)': 'left-x-axis (s)', 'y-axis (s)': 'left-y-axis (s)', 'z-axis (s)': 'left-z-axis (s)' }
        right_columns_names = {'elapsed (s)': 'right-elapsed (s)', 'x-axis (s)': 'right-x-axis (s)', 'y-axis (s)': 'right-y-axis (s)', 'z-axis (s)': 'right-z-axis (s)' }
    else:
        left_columns_names = {'elapsed (s)': 'left-elapsed (s)', 'x-axis (deg/s)': 'left-x-axis (deg/s)', 'y-axis (deg/s)': 'left-y-axis (deg/s)', 'z-axis (deg/s)': 'left-z-axis (deg/s)' }
        right_columns_names = {'elapsed (s)': 'right-elapsed (s)', 'x-axis (deg/s)': 'right-x-axis (deg/s)', 'y-axis (deg/s)': 'right-y-axis (deg/s)', 'z-axis (deg/s)': 'right-z-axis (deg/s)' }
    
    # Rename columns to avoid duplicates
    left_shank_data.rename(columns=left_columns_names, inplace=True)
    left_shank_data.drop(columns=['epoc (ms)'], inplace=True)
    right_shank_data.rename(columns=right_columns_names, inplace=True)
    right_shank_data.drop(columns=['epoc (ms)'], inplace=True)

    # Merge the data on the timestamp, sort by timestamp and convert to datetime
    data = left_shank_data.merge(right_shank_data, on='timestamp (+0700)', how='outer').sort_values('timestamp (+0700)').ffill()
    data['timestamp (+0700)'] = data['timestamp (+0700)'].apply(lambda x: x.replace('T', ' '))
    data['timestamp (+0700)'] = data['timestamp (+0700)'].apply(lambda x: x.replace('.', ':'))
    data['timestamp (+0700)'] = pd.to_datetime(data['timestamp (+0700)'].str.strip(), format='%Y-%m-%d %H:%M:%S:%f')

    data.to_csv(data_dir + '{}.csv'.format(merge_type), index=False)
    


def time_domain_features(data_dir, merge_type):
    '''
    Calculate the metrics for the gyroscope/accelerometer data.
    Args:
        data_dir (str): Directory where the merged gyroscope/accelerometer data is saved.
        merge_type (str): Type of data to be merged (accelerometer or gyroscope).
    '''
    # Check if the files exist
    if not os.path.exists(data_dir + '{}.csv'.format(merge_type)):
        raise FileNotFoundError(f'{merge_type}.csv file not found.')
    
    # Read the merged gyroscope/accelerometer data
    data = pd.read_csv(data_dir + '{}.csv'.format(merge_type))
    data.dropna(inplace=True)
    
    # Calculate the time domain metrics
    # Mean, Standard Deviation, Maximum, Minimum, Root Mean Square, Median Absolute Deviation, Range,
    # Interquartile Range, Skewness & Kurtosis, Zero-crossing rate, Peak count / amplitude
    metrics = {}
    sides = ['left', 'right']
    
    for side in sides:
        metrics[f'{side}-mean']  = data[f'{side}-z-axis (deg/s)'].mean()
        metrics[f'{side}-std']   = data[f'{side}-z-axis (deg/s)'].std()
        metrics[f'{side}-max']   = data[f'{side}-z-axis (deg/s)'].max()
        metrics[f'{side}-min']   = data[f'{side}-z-axis (deg/s)'].min()
        metrics[f'{side}-rms']   = data[f'{side}-z-axis (deg/s)'].apply(lambda x: np.sqrt(np.mean(x**2)))
        metrics[f'{side}-mad']   = data[f'{side}-z-axis (deg/s)'].apply(lambda x: np.median(np.abs(x - np.median(x))))
        metrics[f'{side}-range'] = metrics[f'{side}-max'] - metrics[f'{side}-min']
        metrics[f'{side}-iqr']   = data[f'{side}-z-axis (deg/s)'].apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        metrics[f'{side}-skew']  = data[f'{side}-z-axis (deg/s)'].apply(lambda x: ((x - np.mean(x))**3).mean() / (np.std(x)**3))
        metrics[f'{side}-kurt']  = data[f'{side}-z-axis (deg/s)'].apply(lambda x: ((x - np.mean(x))**4).mean() / (np.std(x)**4))
        metrics[f'{side}-zcr']   = ((data[f'{side}-z-axis (deg/s)'][:-1] * data[f'{side}-z-axis (deg/s)'][1:]) < 0).sum()
        metrics[f'{side}-pkcnt'] = ((data[f'{side}-z-axis (deg/s)'][:-1] * data[f'{side}-z-axis (deg/s)'][1:]) < 0).sum()
        metrics[f'{side}-pkamp'] = data['f{side}-z-axis (deg/s)'].max() - data[f'{side}-z-axis (deg/s)'].min()
    
    # Save the metrics to a CSV file
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(data_dir + f'metrics_{merge_type}.csv'), index=False)
    

def frequency_domain_features(data_dir, merge_type, output_path, fs=100, window_duration_sec=120):
    '''
    Calculate the frequency domain features for the gyroscope data.
    Args:
        data_dir (str): Path to the gyroscope data CSV file.
        merge_type (str): Type of data to be merged (accelerometer or gyroscope).
        output_path (str): Directory where the frequency domain features will be saved.
        fs (int): Sampling frequency in Hz.
        window_duration_sec (int): Duration of the window in seconds.
    '''
    # Check if the files exist
    if not os.path.exists(data_dir + '{}.csv'.format(merge_type)):
        raise FileNotFoundError(f'{merge_type}.csv file not found.')
    
    # Read data
    data = pd.read_csv(data_dir + '{}.csv'.format(merge_type))
    data['timestamp (+0700)'] = pd.to_datetime(data['timestamp (+0700)'])
    data.dropna(inplace=True)

    start_time = data['timestamp (+0700)'].iloc[0]
    end_time = data['timestamp (+0700)'].iloc[-1]
    
    window_features = []
    window_id = 0
    delta = pd.Timedelta(seconds=window_duration_sec)
    current_start = start_time

    # Loop through the data in windows of 2 minutes / 120 seconds and get the important frequency domain features
    # Dominant frequency, Spectral entropy, Gait band energy
    while current_start + delta <= end_time:
        current_end = current_start + delta
        window = data[(data['timestamp (+0700)'] >= current_start) & (data['timestamp (+0700)'] < current_end)]

        for side in ['left', 'right']:
            z_axis = f'{side}-z-axis (deg/s)'
            signal = window[z_axis].values
            n = len(signal)
            if n < 2:
                continue

            fft_vals = fft(signal)
            freqs = fftfreq(n, d=1/fs)
            psd = np.abs(fft_vals)**2

            pos_freqs = freqs[:n // 2]
            pos_psd = psd[:n // 2]

            dominant_freq = pos_freqs[np.argmax(pos_psd)]
            psd_norm = pos_psd / np.sum(pos_psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

            gait_band_energy = np.sum(pos_psd[(pos_freqs >= 0.5) & (pos_freqs <= 3)])

            window_features.append({
                'window_id': window_id,
                'side': side,
                'start_time': current_start,
                'end_time': current_end,
                'dominant_freq': dominant_freq,
                'spectral_entropy': spectral_entropy,
                'gait_band_energy': gait_band_energy,
                'samples': n
            })
        window_id += 1
        current_start += delta

    features_df = pd.DataFrame(window_features)
    features_df.to_csv(os.path.join(output_path, 'windowed_frequency_features.csv'), index=False)

    
def gait_features(data_dir, merge_type):
    '''
    Calculate the gait features for the gyroscope data.
    Args:
        data_dir (str): Directory where the merged gyroscope data is saved.
        merge_type (str): Type of data to be merged (accelerometer or gyroscope).
    '''
    # Check if the files exist
    if not os.path.exists(data_dir + '{}.csv'.format(merge_type)):
        raise FileNotFoundError(f'{merge_type}.csv file not found.')
    
    # Read the merged gyroscope/accelerometer data
    data = pd.read_csv(data_dir + '{}.csv'.format(merge_type))
    data.dropna(inplace=True)
    
    
    # Calculate the gait metrics including stride times, stance/swing times, asymmetry index, and symmetry ratio
    left_peaks = signal.find_peaks(data['left-z-axis (deg/s)'], height=0.5, distance=100)
    right_peaks = signal.find_peaks(data['right-z-axis (deg/s)'], height=0.5, distance=100)
    left_stride_times = np.diff(left_peaks[0])
    right_stride_times = np.diff(right_peaks[0])
    
    left_stance_swing = detect_stance_swing(data['left-z-axis (deg/s)'], data['timestamp (+0700)'])
    right_stance_swing = detect_stance_swing(data['right-z-axis (deg/s)'], data['timestamp (+0700)'])
    
    asymmetry = asymmetry_index(left_stride_times, right_stride_times)
    symmetry = symmetry_ratio(left_stride_times, right_stride_times)
    gait_metrics = {
        'left_stride_times': left_stride_times,
        'right_stride_times': right_stride_times,
        'left_stance_swing': left_stance_swing,
        'right_stance_swing': right_stance_swing,
        'asymmetry_index': asymmetry,
        'symmetry_ratio': symmetry
    }
    
    # Save the gait metrics to a CSV file
    gait_metrics_df = pd.DataFrame(gait_metrics)
    gait_metrics_df.to_csv(os.path.join(data_dir, 'gait_metrics.csv'), index=False)
    


def cross_limb_features(data_dir, merge_type, fs=100):
    # Check if the files exist
    if not os.path.exists(data_dir + '{}.csv'.format(merge_type)):
        raise FileNotFoundError(f'{merge_type}.csv file not found.')
    
    # Read the merged gyroscope/accelerometer data
    data = pd.read_csv(data_dir + '{}.csv'.format(merge_type))
    data.dropna(inplace=True)
    
    
    data['left_z_filtered'] = butter_low_pass(data['left-z-axis (deg/s)'], fs=fs)
    data['right_z_filtered'] = butter_low_pass(data['right-z-axis (deg/s)'], fs=fs)
    
    left_peaks, _ = find_peaks(data['left_z_filtered'], distance=fs*0.5)
    right_peaks, _ = find_peaks(data['right_z_filtered'], distance=fs*0.5)
    
    features = []
    for i in range(min(len(left_peaks), len(right_peaks)) - 1):
        l_start, l_end = left_peaks[i], left_peaks[i+1]
        r_start, r_end = right_peaks[i], right_peaks[i+1]

        l_cycle = data.iloc[l_start:l_end]
        r_cycle = data.iloc[r_start:r_end]

        # Truncate to shortest cycle
        min_len = min(len(l_cycle), len(r_cycle))
        if min_len < 5:
            continue  # skip very short cycles to avoid noise

        left_stride_duration  = data['left-elapsed (s)'].iloc[l_end]  - data['left-elapsed (s)'].iloc[l_start]
        right_stride_duration = data['right-elapsed (s)'].iloc[r_end] - data['right-elapsed (s)'].iloc[r_start]

        feature = {
            "left_stride_duration": left_stride_duration,
            "right_stride_duration": right_stride_duration,
            "stride_duration_diff": abs(left_stride_duration - right_stride_duration),
            "stride_duration_symmetry_ratio": min(left_stride_duration, right_stride_duration) / max(left_stride_duration, right_stride_duration),
            "left_peak": data['left_z_filtered'].iloc[l_start:l_end].max(),
            "right_peak": data['right_z_filtered'].iloc[r_start:r_end].max(),
            "peak_diff": abs(data['left_z_filtered'].iloc[l_start:l_end].max() - df['right_z_filtered'].iloc[r_start:r_end].max()),
            "z_corr": np.corrcoef(
                l_cycle['left_z_filtered'].values[:min_len],
                r_cycle['right_z_filtered'].values[:min_len]
            )[0,1]
        }
        features.append(feature)

    # Save the cross limb metrics to a CSV file
    cross_limb_features = pd.DataFrame(features)
    cross_limb_features.to_csv(os.path.join(data_dir, 'cross_limb_metrics.csv'), index=False)


def butter_low_pass(data, cutoff=6, fs=100, order=2):
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


def detect_stance_swing(z_filtered, time):
    '''
    Detect stance and swing phases from the gyroscope data.
    Args:
        z_filtered (np.array): Filtered z-axis gyroscope data.
        time (np.array): Corresponding time values.
    '''
    zero_crossings = np.where(np.diff(np.sign(z_filtered)))[0]
    phases = []
    for i in range(len(zero_crossings) - 1):
        start = zero_crossings[i]
        end = zero_crossings[i+1]
        min_idx = np.argmin(z_filtered[start:end]) + start
        phases.append({
            "stance_time": time[min_idx] - time[start],
            "swing_time": time[end] - time[min_idx]
        })
    return phases

def asymmetry_index(left, right):
    '''
    Calculate the asymmetry index between left and right stride times.
    Args:
        left (list): Left stride times.
        right (list): Right stride times.
    '''
    return [(l - r) / (l + r) if (l + r) != 0 else 0 for l, r in zip(left, right)]

def symmetry_ratio(left, right):
    '''
    Calculate the symmetry ratio between left and right stride times.
    Args:
        left (list): Left stride times.
        right (list): Right stride times.
    '''
    return [min(l, r) / max(l, r) if max(l, r) != 0 else 0 for l, r in zip(left, right)]