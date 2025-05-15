patient_path = 'Data/Healthy/Patient_1'



patiend_id = patient_path.split('/')[-1].lower()
status = 0 if patient_path.__contains__('Healthy') else 1
patient_id = patiend_id + '_' + str(status)
print(patient_id)