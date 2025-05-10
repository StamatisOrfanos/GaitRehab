from data_preprocessing import merge_data
merge_data('', 'Data/Healthy/Patient_3/LeftShank-Accelerometer.csv', 'Data/Healthy/Patient_3/RightShank-Accelerometer.csv', 'accelerometer')
merge_data('', 'Data/Healthy/Patient_3/LeftShank-Gyroscope.csv', 'Data/Healthy/Patient_3/RightShank-Gyroscope.csv', 'gyroscope')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load and prepare data
accel_df = pd.read_csv("accelerometer.csv")
gyro_df = pd.read_csv("gyroscope.csv")
accel_df["timestamp"] = pd.to_datetime(accel_df["timestamp (+0700)"])
gyro_df["timestamp"] = pd.to_datetime(gyro_df["timestamp (+0700)"])

# Merge by timestamp
merged_df = pd.merge_asof(
    gyro_df.sort_values("timestamp"),
    accel_df.sort_values("timestamp"),
    on="timestamp",
    direction="nearest"
)

# Limit to a 10-second window
start_time = merged_df["timestamp"].iloc[0]
plot_range = merged_df[
    (merged_df["timestamp"] >= start_time) &
    (merged_df["timestamp"] <= start_time + pd.Timedelta(seconds=10))
]

# Extract signals for right leg
time = plot_range["timestamp"]
gyro_z = plot_range["right-z-axis (deg/s)"]
accel_z = plot_range["right-z-axis (g)"]


from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

# Reload gyroscope signal for the right leg
gyro_df = pd.read_csv("gyroscope.csv")
gyro_df["timestamp"] = pd.to_datetime(gyro_df["timestamp (+0700)"])
gyro_z = gyro_df["right-z-axis (deg/s)"].values
time = gyro_df["timestamp"].values

# Find heel strikes based on sharp negative peaks (simulated here using minima in gyro Z)
peaks, _ = find_peaks(-gyro_z, distance=50, prominence=10)  # Simulated heel strikes

# Extract stride segments and normalize to 100 points
normalized_strides = []

for i in range(len(peaks) - 1):
    start = peaks[i]
    end = peaks[i + 1]
    segment = gyro_z[start:end]
    
    if len(segment) < 10:
        continue

    x_old = np.linspace(0, 100, len(segment))
    f_interp = interp1d(x_old, segment, kind="linear")
    x_new = np.linspace(0, 100, 100)
    normalized = f_interp(x_new)
    normalized_strides.append(normalized)

# Convert to array and compute mean and std
strides_array = np.vstack(normalized_strides)
mean_curve = np.mean(strides_array, axis=0)
std_curve = np.std(strides_array, axis=0)
x = np.linspace(0, 100, 100)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(x, mean_curve, color='blue', linewidth=2, label="Average Angular Velocity")
plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color='blue', alpha=0.2, label="±1 SD")

# Simulated event markers
plt.scatter([0], [mean_curve[0]], color='red', edgecolor='black', s=100, label="Heel Strike", zorder=5)
plt.scatter([60], [mean_curve[60]], color='red', marker='*', s=150, label="Toe Off", zorder=5)

plt.axvline(60, color='black', linestyle='--', alpha=0.5)
plt.title("Normal Person", fontsize=14, weight='bold')
plt.xlabel("Time (% of gait cycle)")
plt.ylabel("Angular velocity (deg/s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# -----------------------------------------------------------------------------------------------------------------------------


# # Detect peaks in gyroscope (mid-swing)
# gyro_peaks, _ = find_peaks(gyro_z, distance=10, height=np.percentile(gyro_z, 75))

# # Detect spikes in accelerometer (heel strike / toe-off)
# accel_peaks, _ = find_peaks(accel_z, distance=10, height=np.percentile(accel_z, 90))

# # Plot
# plt.figure(figsize=(15, 6))
# plt.plot(time, gyro_z, label="Gyroscope Z (deg/s)", color="blue")
# plt.plot(time, accel_z, label="Accelerometer Z (g)", color="red", alpha=0.7)
# plt.scatter(time.iloc[gyro_peaks], gyro_z.iloc[gyro_peaks], color='blue', marker='o', label="Mid-Swing (Gyro Peak)")
# plt.scatter(time.iloc[accel_peaks], accel_z.iloc[accel_peaks], color='red', marker='x', label="Heel Strike/Toe-Off (Accel Spike)")
# plt.title("Right Leg - Gait Cycle Events")
# plt.xlabel("Time")
# plt.ylabel("Signal")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# from scipy.signal import find_peaks

# # Re-use merged_df from previous step
# # Define right leg signals
# time = merged_df["timestamp"]
# gyro_z = merged_df["right-z-axis (deg/s)"]
# accel_z = merged_df["right-z-axis (g)"]

# # Step 1: Detect heel strikes from accelerometer Z (sharp impact peaks)
# heel_strike_indices, _ = find_peaks(accel_z, distance=10, height=np.percentile(accel_z, 90))

# # Step 2: Detect toe-offs using valleys (minimums) in gyro Z after heel strike
# toe_off_indices = []
# for hs_idx in heel_strike_indices[:-1]:  # avoid last point overflow
#     next_hs_idx = heel_strike_indices[np.where(heel_strike_indices > hs_idx)[0][0]]
#     segment = gyro_z.iloc[hs_idx:next_hs_idx]
#     if len(segment) > 1:
#         toe_off_idx = segment.idxmin()
#         toe_off_indices.append(toe_off_idx)

# # Step 3: Plot with swing (toe-off → heel strike) and stance (heel strike → toe-off)
# plt.figure(figsize=(15, 6))
# plt.plot(time, gyro_z, label="Gyroscope Z (deg/s)", color="blue")
# plt.plot(time, accel_z, label="Accelerometer Z (g)", color="red", alpha=0.7)

# # Heel strike and toe-off markers
# plt.scatter(time.iloc[heel_strike_indices], accel_z.iloc[heel_strike_indices], color='red', marker='x', label="Heel Strike (Accel Spike)")
# plt.scatter(time.loc[toe_off_indices], gyro_z.loc[toe_off_indices], color='green', marker='o', label="Toe Off (Gyro Valley)")

# # Shade swing and stance intervals
# for i in range(min(len(heel_strike_indices)-1, len(toe_off_indices))):
#     hs_idx = heel_strike_indices[i]
#     to_idx = toe_off_indices[i]
#     hs_time = time.iloc[hs_idx]
#     to_time = time.loc[to_idx]

#     if hs_time < to_time:
#         plt.axvspan(hs_time, to_time, color='orange', alpha=0.2, label='Stance Phase' if i == 0 else "")
#         next_hs_time = time.iloc[heel_strike_indices[i+1]]
#         plt.axvspan(to_time, next_hs_time, color='green', alpha=0.2, label='Swing Phase' if i == 0 else "")

# plt.title("Gait Cycle: Heel Strike and Toe Off Detection")
# plt.xlabel("Time")
# plt.ylabel("Signal")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
