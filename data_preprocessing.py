# Libraries
import os
import shutil
import numpy as np
import pandas as pd


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



def aggregate_features(base_dir, label):
    '''
    Aggregate features from multiple CSV files for each patient.
    Args:
        base_dir (str): Base directory containing patient folders.
        label (int): Label for the dataset (0 for healthy, 1 for stroke).
    '''
    all_features = []

    for patient_id in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue

        patient_features = {'subject_id': patient_id}

        try:
            # Process individual feature types
            process_time_domain_features(patient_path, patient_features)
            process_frequency_domain_features(patient_path, patient_features)
            process_gait_summary_metrics(patient_path, patient_features)
            process_cross_limb_metrics(patient_path, patient_features)

            # Add Label, 0 = healthy, 1 = stroke
            patient_features['label'] = label
            all_features.append(patient_features)

        except Exception as e:
            print(f'Error processing {patient_id}: {e}')

    return pd.DataFrame(all_features)


def process_time_domain_features(patient_path, patient_features):
    '''
    Get time domain features from the gyroscope data.
    Args:
        patient_path (str): Path to the patient's data directory.
        patient_features (dict): Dictionary to store the features.
    '''
    time_path = os.path.join(patient_path, 'time_domain_metrics_gyroscope.csv')
    if os.path.exists(time_path):
        time_df = pd.read_csv(time_path)
        for col in time_df.columns:
            patient_features[col] = time_df[col].iloc[0]


def process_frequency_domain_features(patient_path, patient_features):
    '''
    Get frequency domain features from the gyroscope data.
    Args:
        patient_path (str): Path to the patient's data directory.
        patient_features (dict): Dictionary to store the features.
    '''
    freq_path = os.path.join(patient_path, 'windowed_frequency_features_gyroscope.csv')
    if os.path.exists(freq_path):
        freq_df = pd.read_csv(freq_path)
        grouped = freq_df.groupby('side')
        for side in ['left', 'right']:
            if side in grouped.groups:
                side_df = grouped.get_group(side)
                for feat in ['dominant_freq', 'spectral_entropy', 'gait_band_energy']:
                    patient_features[f'{side}_{feat}_mean'] = side_df[feat].mean()
                    patient_features[f'{side}_{feat}_std'] = side_df[feat].std()


def process_gait_summary_metrics(patient_path, patient_features):
    '''
    Get gait summary metrics from the gyroscope data.
    Args:
        patient_path (str): Path to the patient's data directory.
        patient_features (dict): Dictionary to store the features.
    '''
    gait_summary_path = os.path.join(patient_path, 'gait_features', 'summary_gait_metrics_gyroscope.csv')
    if os.path.exists(gait_summary_path):
        gait_df = pd.read_csv(gait_summary_path)
        for col in gait_df.columns:
            patient_features[col] = gait_df[col].mean()


def process_cross_limb_metrics(patient_path, patient_features):
    '''
    Get cross-limb metrics from the gyroscope data.
    Args:
        patient_path (str): Path to the patient's data directory.
        patient_features (dict): Dictionary to store the features.
    '''
    cross_path = os.path.join(patient_path, 'cross_limb_metrics.csv')
    if os.path.exists(cross_path):
        cross_df = pd.read_csv(cross_path)
        for col in cross_df.columns:
            if col not in ['window_id', 'start_time', 'end_time', 'side']:
                patient_features[f'{col}_mean'] = cross_df[col].mean()
                patient_features[f'{col}_std'] = cross_df[col].std()


def clean_all_patients(base_dir='Healthy', data_type='gyroscope'):
    '''
    Loop over all subdirectories in base_dir and delete feature files.
    Args:
        base_dir (str): Base directory containing patient folders.
        data_type (str): Type of data to be cleaned (accelerometer or gyroscope).
    '''
    for patient_id in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_id)
        if os.path.isdir(patient_path):
            delete_feature_files(patient_path, data_type=data_type)

    print(f'\n Cleanup complete for all patients in {base_dir}')


def delete_feature_files(patient_dir, data_type='gyroscope'):
    '''
    Deletes all feature CSVs generated inside a patient's folder.
    Args:
        patient_dir (str): Directory of the patient.
        data_type (str): Type of data to be cleaned (accelerometer or gyroscope).
    '''
    # Files in the root directory
    root_files = [
        f'time_domain_metrics_{data_type}.csv',
        f'windowed_frequency_features_{data_type}.csv',
        'cross_limb_metrics.csv',
        f'{data_type}.csv'
    ]
    
    for f in root_files:
        path = os.path.join(patient_dir, f)
        if os.path.exists(path):
            os.remove(path)
            print(f'Deleted {f} in {patient_dir}')
    
    # Delete gait_features directory
    gait_dir = os.path.join(patient_dir, 'gait_features')
    if os.path.exists(gait_dir):
        shutil.rmtree(gait_dir)
        print(f'Deleted gait_features/ in {patient_dir}')