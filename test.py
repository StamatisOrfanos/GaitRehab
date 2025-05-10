from data_preprocessing import merge_all_types, merge_data


merge_data('Data/Healthy', 'Data/Healthy/Patient_1/LeftShank-Accelerometer.csv', 'Data/Healthy/Patient_1/RightShank-Accelerometer.csv', 'accelerometer')
merge_data('Data/Healthy', 'Data/Healthy/Patient_1/LeftShank-Gyroscope.csv', 'Data/Healthy/Patient_1/RightShank-Gyroscope.csv', 'gyroscope')