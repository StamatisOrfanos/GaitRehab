# Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Display settings
plt.rcParams['figure.figsize'] = (15, 5)

# Constants
data_path = "/Users/stamatiosorphanos/Documents/GaitRehab_Models/data"
healthy_directory = os.path.join(data_path, "Healthy")

stroke_directory  = os.path.join(data_path, "Stroke")

positions_dict = {}



# Renaming and standardizing the files
for subdir, dirs, files in os.walk(healthy_directory):
    for file in files:
        file_path = os.path.join(healthy_directory, file)
        formatted_name = file.replace(" ", "").replace("-", "_")
        sensors_position = formatted_name.split("_")[0]
        device = formatted_name.split("_")[-1].split(".")[0]
        date = "/".join(formatted_name.split("T")[0].split("_")[1:4])
        print(subdir + "\t" + device + "\t" + sensors_position + "\t" + date)
        new_name = file.split("_")[0] + ".csv"