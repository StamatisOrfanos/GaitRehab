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