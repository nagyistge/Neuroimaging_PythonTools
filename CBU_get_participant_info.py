import dicom
import os
import pandas as pd
import re

results = []
in_folder = '/mridata/cbu/'
CALM_files = sorted([folder for folder in os.listdir(in_folder) if re.search('_CALM_', folder)])

for CALM_file in CALM_files:
    filename_base = in_folder + CALM_file + '/' + os.listdir(in_folder + CALM_file)[0] + '/'
    folder = filename_base + 'Series_005_CBU_MPRAGE_32chn'

    if not os.path.isdir(folder):
        folder = filename_base + 'Series_014_CBU_TSE_32chn'

    ID = CALM_file.split('_')[0]

    if os.path.isdir(folder):
        header = dicom.read_file(folder + '/' + os.listdir(folder)[0])
        height = header.PatientSize
        weight = header.PatientWeight
        gender = header.PatientSex
    else:
        height = weight = gender = float('nan')

    results.append({'MRI.ID': ID, 'height': height, 'weight':weight, 'gender': gender})

pd.DataFrame(results).to_csv('CALM_MRI_participant_info.csv')
