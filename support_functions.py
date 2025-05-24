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

def summarize(lst: list):
    '''
    Calculate the mean of a list.
    Args:
        lst (list): List of values.
    '''
    return float(np.mean(lst)) if len(lst) > 0 else 0

def simulate_gait_signal(amplitude: float, freq: float, phase_shift: float, noise: float, duration_sec=2.0, fs=100):
    '''
    Simulate a gait signal with noise.
    Args:
        amplitude (float): Amplitude of the sine wave.
        freq (float): Frequency of the sine wave.
        phase_shift (float): Phase shift of the sine wave.
        noise (float): Standard deviation of the Gaussian noise.
        duration_sec (float): Duration of the signal in seconds.
        fs (int): Sampling frequency in Hz.
    '''
    t = np.linspace(0, duration_sec, int(fs * duration_sec))
    signal = amplitude * np.sin(2 * np.pi * freq * t + phase_shift)
    return signal + np.random.normal(0, noise, size=t.shape)