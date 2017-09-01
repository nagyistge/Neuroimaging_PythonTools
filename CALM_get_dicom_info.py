import os
import re
import dicom
import pandas as pd

folder = '/mridata/cbu/'
IDs = list()
Heights = list()
Weights = list()
Sexs = list()
DOBs = list()
ScanDates = list()

for subfolder in os.listdir(folder):
	if re.search('CALM_STRUCTURAL', subfolder):
		info = dicom.read_file(folder + subfolder + '/' + os.listdir(folder + subfolder)[0] + '/' + 
			os.listdir(folder + subfolder + '/' + os.listdir(folder + subfolder)[0])[0] + '/' +
			os.listdir(folder + subfolder + '/' + os.listdir(folder + subfolder)[0] + '/' + 
			os.listdir(folder + subfolder + '/' + os.listdir(folder + subfolder)[0])[0])[0])

		IDs.append(info.PatientName)
		Heights.append(info.PatientSize)
		Weights.append(info.PatientWeight)
		Sexs.append(info.PatientSex)
		DOBs.append(info.PatientBirthDate)
		ScanDates.append(info.AcquisitionDate)

df = pd.DataFrame({'D.o.B': DOBs,
				  'Date of Scan': ScanDates,
				  'Gender': Sexs,
				  'Height [m]': Heights,
				  'Weight [kg]': Weights},
				  index = IDs)

df.to_csv('/home/jb07/Desktop/CALM_height_and_weight.csv')