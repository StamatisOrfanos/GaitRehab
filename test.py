import pandas as pd
import os
from datetime import datetime, timedelta

def fix_format(path):
    df = pd.read_csv(path, index_col=0)

    # Rename columns
    df = df.rename(columns={'Gyro1x': 'x-axis (deg/s)', 'Gyro1y': 'y-axis (deg/s)', 'Gyro1z': 'z-axis (deg/s)', 
                            'Time': 'timestamp (+0700)', 'microsecond': 'micro', 'elasped_time': 'elapsed (s)'})

    df["elapsed (s)"]       = (df["micro"] - df["micro"].iloc[0]) / 1000
    base_datetime           = datetime.strptime("2024-08-21 09:35:00.000", "%Y-%m-%d %H:%M:%S.%f")
    df["timestamp (+0700)"] = df["elapsed (s)"].apply(lambda s: base_datetime + timedelta(seconds=s))
    df["epoc (ms)"]         = df["timestamp (+0700)"].apply(lambda x: x.timestamp() * 1000)
    df['x-axis (deg/s)']    = df['x-axis (deg/s)'].round(3)
    df['y-axis (deg/s)']    = df['y-axis (deg/s)'].round(3)
    df['z-axis (deg/s)']    = df['z-axis (deg/s)'].round(3)


    df_order = df[['epoc (ms)', 'timestamp (+0700)', 'elapsed (s)', 'x-axis (deg/s)', 'y-axis (deg/s)', 'z-axis (deg/s)']]
    df_order.to_csv('LeftShank-Gyroscope.csv', index=False)