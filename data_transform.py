# Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def merge_data(data_dir, left_shank_path, right_shank_path, merge_type):
    """
    Merge gyroscope data from left and right shank.
    Args:
        data_dir (str): Directory where the merged gyroscope data will be saved.
        left_shank_path (str): Path to the left shank gyroscope CSV file.
        right_shank_path (str): Path to the right shank gyroscope CSV file.
        merge_type (str): Type of data to be merged (accelerometer or gyroscope).
    """
    # Check if the files exist
    if not os.path.exists(left_shank_path):
        raise FileNotFoundError("LeftShank-Gyroscope.csv file not found.")
    if not os.path.exists(right_shank_path):
        raise FileNotFoundError("RightShank-Gyroscope.csv file not found.")

    # Read the CSV files
    left_shank_data  = pd.read_csv(left_shank_path)
    right_shank_data = pd.read_csv(right_shank_path)
    
    if merge_type == "accelerometer":
        left_columns_names = {'elapsed (s)': 'left-elapsed (s)', "x-axis (s)": "left-x-axis (s)", "y-axis (s)": "left-y-axis (s)", "z-axis (s)": "left-z-axis (s)" }
        right_columns_names = {'elapsed (s)': 'right-elapsed (s)', "x-axis (s)": "right-x-axis (s)", "y-axis (s)": "right-y-axis (s)", "z-axis (s)": "right-z-axis (s)" }
    else:
        left_columns_names = {'elapsed (s)': 'left-elapsed (s)', "x-axis (deg/s)": "left-x-axis (deg/s)", "y-axis (deg/s)": "left-y-axis (deg/s)", "z-axis (deg/s)": "left-z-axis (deg/s)" }
        right_columns_names = {'elapsed (s)': 'right-elapsed (s)', "x-axis (deg/s)": "right-x-axis (deg/s)", "y-axis (deg/s)": "right-y-axis (deg/s)", "z-axis (deg/s)": "right-z-axis (deg/s)" }
    
    # Rename columns to avoid duplicates
    left_shank_data.rename(columns=left_columns_names, inplace=True)
    left_shank_data.drop(columns=['epoc (ms)'], inplace=True)
    right_shank_data.rename(columns=right_columns_names, inplace=True)
    right_shank_data.drop(columns=['epoc (ms)'], inplace=True)

    # Merge the data on the timestamp, sort by timestamp and convert to datetime
    data = left_shank_data.merge(right_shank_data, on="timestamp (+0700)", how="outer").sort_values("timestamp (+0700)").ffill()
    data["timestamp (+0700)"] = data["timestamp (+0700)"].apply(lambda x: x.replace("T", " "))
    data["timestamp (+0700)"] = data["timestamp (+0700)"].apply(lambda x: x.replace(".", ":"))
    data["timestamp (+0700)"] = pd.to_datetime(data['timestamp (+0700)'].str.strip(), format="%Y-%m-%d %H:%M:%S:%f")

    data.to_csv(data_dir + '{}.csv'.format(merge_type), index=False)
    


def time_domain_features(data_dir, merge_type):
    """
    Calculate the metrics for the gyroscope data.
    Args:
        data_dir (str): Directory where the merged gyroscope data is saved.
        merge_type (str): Type of data to be merged (accelerometer or gyroscope).
    """
    # Check if the files exist
    if not os.path.exists(data_dir + '{}.csv'.format(merge_type)):
        raise FileNotFoundError(f"{merge_type}.csv file not found.")
    
    # Read the merged gyroscope data
    gyroscope = pd.read_csv(data_dir + '{}.csv'.format(merge_type))
    gyroscope.dropna(inplace=True)
    
    # Calculate the metrics
    # Mean, Standard Deviation, Maximum, Minimum, Root Mean Square, Median Absolute Deviation, Range,
    # Interquartile Range, Skewness & Kurtosis, Zero-crossing rate, Peak count / amplitude
    metrics = {}
    
    # Right Shank
    metrics['right-mean'] = gyroscope["right-z-axis (deg/s)"].mean()
    metrics['right-std']  = gyroscope["right-z-axis (deg/s)"].std()
    metrics['right-max']  = gyroscope["right-z-axis (deg/s)"].max()
    metrics['right-min']  = gyroscope["right-z-axis (deg/s)"].min()
    metrics['right-rms']   = gyroscope["right-z-axis (deg/s)"].apply(lambda x: np.sqrt(np.mean(x**2)))
    metrics['right-mad']   = gyroscope["right-z-axis (deg/s)"].apply(lambda x: np.median(np.abs(x - np.median(x))))
    metrics['right-range'] = metrics['right-max'] - metrics['right-min']
    metrics['right-iqr']   = gyroscope["right-z-axis (deg/s)"].apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    metrics['right-skew']  = gyroscope["right-z-axis (deg/s)"].apply(lambda x: ((x - np.mean(x))**3).mean() / (np.std(x)**3))
    metrics['right-kurt']  = gyroscope["right-z-axis (deg/s)"].apply(lambda x: ((x - np.mean(x))**4).mean() / (np.std(x)**4))
    metrics['right-zcr']   = ((gyroscope["right-z-axis (deg/s)"][:-1] * gyroscope["right-z-axis (deg/s)"][1:]) < 0).sum()
    metrics['right-pkcnt'] = ((gyroscope["right-z-axis (deg/s)"][:-1] * gyroscope["right-z-axis (deg/s)"][1:]) < 0).sum()
    metrics['right-pkamp'] = gyroscope["right-z-axis (deg/s)"].max() - gyroscope["right-z-axis (deg/s)"].min()
    
    
    # Left Shank
    metrics['left-mean'] = gyroscope["left-z-axis (deg/s)"].mean()
    metrics['left-std']  = gyroscope["left-z-axis (deg/s)"].std()
    metrics['left-max']  = gyroscope["left-z-axis (deg/s)"].max()
    metrics['left-min']  = gyroscope["left-z-axis (deg/s)"].min()
    metrics['left-rms']  = gyroscope["left-z-axis (deg/s)"].apply(lambda x: np.sqrt(np.mean(x**2)))    
    metrics['left-rms']   = gyroscope["left-z-axis (deg/s)"].apply(lambda x: np.sqrt(np.mean(x**2)))
    metrics['left-mad']   = gyroscope["left-z-axis (deg/s)"].apply(lambda x: np.median(np.abs(x - np.median(x))))
    metrics['left-range'] = metrics['left-max'] - metrics['left-min']
    metrics['left-iqr']   = gyroscope["left-z-axis (deg/s)"].apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    metrics['left-skew']  = gyroscope["left-z-axis (deg/s)"].apply(lambda x: ((x - np.mean(x))**3).mean() / (np.std(x)**3))
    metrics['left-kurt']  = gyroscope["left-z-axis (deg/s)"].apply(lambda x: ((x - np.mean(x))**4).mean() / (np.std(x)**4))
    metrics['left-zcr']   = ((gyroscope["left-z-axis (deg/s)"][:-1] * gyroscope["left-z-axis (deg/s)"][1:]) < 0).sum()
    metrics['left-pkcnt'] = ((gyroscope["left-z-axis (deg/s)"][:-1] * gyroscope["left-z-axis (deg/s)"][1:]) < 0).sum()
    metrics['left-pkamp'] = gyroscope["left-z-axis (deg/s)"].max() - gyroscope["left-z-axis (deg/s)"].min()
    
    # Save the metrics to a CSV file
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(data_dir + f'metrics_{merge_type}.csv'), index=False)
    
    
    
    
    
    