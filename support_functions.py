# Libraries
import numpy as np
import pandas as pd



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

def summarize(lst):
    return float(np.mean(lst)) if len(lst) > 0 else 0

def simulate_gait_signal(amplitude, freq, phase_shift, noise, duration_sec=2.0, fs=100):
    t = np.linspace(0, duration_sec, int(fs * duration_sec))
    signal = amplitude * np.sin(2 * np.pi * freq * t + phase_shift)
    return signal + np.random.normal(0, noise, size=t.shape)