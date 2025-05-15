# Libraries
import os
import shutil
import numpy as np
import pandas as pd

# Merge gyroscope data from left and right shank ---------------------------------------------

def merge_data(data_dir: str, left_shank_path: str, right_shank_path: str, merge_type: str):
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
        raise FileNotFoundError(f'LeftShank-{merge_type}.csv file not found.')
    if not os.path.exists(right_shank_path):
        raise FileNotFoundError(f'RightShank-{merge_type}.csv file not found.')

    # Read the CSV files
    left_shank_data  = pd.read_csv(left_shank_path)
    right_shank_data = pd.read_csv(right_shank_path)
        
    if merge_type == 'accelerometer':
        left_columns_names = {'elapsed (s)': 'left-elapsed (s)', 'x-axis (g)': 'left-x-axis (g)', 'y-axis (g)': 'left-y-axis (g)', 'z-axis (g)': 'left-z-axis (g)' }
        right_columns_names = {'elapsed (s)': 'right-elapsed (s)', 'x-axis (g)': 'right-x-axis (g)', 'y-axis (g)': 'right-y-axis (g)', 'z-axis (g)': 'right-z-axis (g)' }
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

    data.to_csv(data_dir + '/' + '{}.csv'.format(merge_type), index=False)



# Aggregate features from all the features files ---------------------------------------------


#  Classification Functions ----------------------------

def classification_merge_all_types(health_dir: str, stroke_dir: str):
    '''
    Merge all types of data for each patient.
    Args:
        health_dir (str): Directory containing healthy patients' data.
        stroke_dir (str): Directory containing stroke patients' data.
    '''
    output_path = 'final_dataset.csv'
    healthy_df  = pd.read_csv(os.path.join(health_dir, 'healthy_features.csv'))
    stroke_df   = pd.read_csv(os.path.join(stroke_dir, 'stroke_features.csv'))
    
    # Combine both datasets
    full_df = pd.concat([healthy_df, stroke_df], ignore_index=True)
    full_df.to_csv(output_path, index=False)
    full_df.to_csv(output_path, index=False)
    print(f'Saved dataset with shape {full_df.shape} to {output_path}')


def classification_aggregate_features(base_dir: str, label: int):
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


def process_time_domain_features(patient_path: str, patient_features: dict):
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


def process_frequency_domain_features(patient_path: str, patient_features: dict):
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


def process_gait_summary_metrics(patient_path: str, patient_features: dict):
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


def process_cross_limb_metrics(patient_path: str, patient_features:dict):
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

       
# Detection Functions ----------------------------

def detection_merge_subject_features(base_dir: str, filename: str, output_name: str):
    """
    Merge per-subject detection feature CSVs into a single file for training.
    Args:
        base_dir (str): Root directory containing subjects subfolders.
        filename (str): Filename to look for in each subjects folder.
        output_name (str): Output CSV file name to save the merged result.
    """
    all_dfs = []

    for patient_folder in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_folder)
        file_path = os.path.join(patient_path, filename)

        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            df['patient_id'] = patient_folder
            all_dfs.append(df)

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.to_csv(os.path.join(base_dir, output_name), index=False)
        print(f"Merged {len(all_dfs)} files into {output_name}")
    else:
        print("No files found to merge.")
        

def detection_merge_raw_npz_files(base_dir: str, filename="detection_raw_window.npz", output_name="all_subject_raw_windows.npz"):
    '''
    Merge all npz data from all the subjects into a single file for training
    Args:
        base_dir (str): Root directory containing patient subfolders.
        filename (str): Filename to look for in each patient folder.
        output_name (str): Output CSV file name to save the merged result.
    '''
    all_arrays = []

    for patient_folder in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_folder)
        npz_path = os.path.join(patient_path, filename)

        if os.path.exists(npz_path):
            data = np.load(npz_path)
            all_arrays.append(data["X"])

    if all_arrays:
        merged = np.concatenate(all_arrays, axis=0)
        np.savez_compressed(os.path.join(base_dir, output_name), X=merged)
        print(f"Merged {len(all_arrays)} files into {output_name} with shape {merged.shape}")
    else:
        print("No .npz files found.")


def detection_merge_csv_datasets(health_dir: str, stroke_dir: str, file_type: str):
    '''
    Merge all types of data for each patient/subject.
    Args:
        health_dir (str): Directory containing healthy subjects' data.
        stroke_dir (str): Directory containing stroke patients' data.
        file_type (str): Type of file we are merging [detection_time_domain, detection_asymmetry]
    '''
    output_path = f'{file_type}.csv'
    healthy_df  = pd.read_csv(os.path.join(health_dir, f'{file_type}.csv'))
    stroke_df   = pd.read_csv(os.path.join(stroke_dir, f'{file_type}.csv'))
    
    # Combine both datasets
    full_df = pd.concat([healthy_df, stroke_df], ignore_index=True)
    full_df.to_csv(output_path, index=False)
    full_df.to_csv(output_path, index=False)
    print(f'Saved dataset {file_type} with shape {full_df.shape} to {output_path}')

 
def detection_merge_npz_files(base_dir: str, filename="detection_raw_window.npz", output_name="all_subject_raw_windows.npz"):
    '''
    Merge all types of data for each patient/subject.
    Args:
        base_dir (str): Directory containing healthy subjects' data.
    '''
    all_arrays = []

    for base_folders in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, base_folders)
        npz_path = os.path.join(patient_path, filename)

        if os.path.exists(npz_path):
            data = np.load(npz_path)
            all_arrays.append(data["X"])

    if all_arrays:
        merged = np.concatenate(all_arrays, axis=0)
        np.savez_compressed(os.path.join(base_dir, output_name), X=merged)
        print(f"Merged {len(all_arrays)} files into {output_name} with shape {merged.shape}")
    else:
        print("No .npz files found.")  


# Clean up function to delete all feature files ----------------------------

def clean_extra_files(base_dir: str, data_type='gyroscope'):
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


def delete_feature_files(patient_dir: str, data_type='gyroscope'):
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
        f'{data_type}.csv',
        'detection_time_domain.csv',
        'detection_asymmetry.csv',
        'detection_raw_window.npz'
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