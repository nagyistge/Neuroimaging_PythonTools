#! /usr/bin/env python
import sys
import optparse

def main():
	p = optparse.OptionParser()
	p.add_option('--source_folder')
	p.add_option('--target_folder')
	p.add_option('--csv')
	options, arguments = p.parse_args()
	source_folder = options.source_folder
	target_folder = options.target_folder
	csv = options.csv

	import os 
	if not os.path.isdir(target_folder):
		os.mkdir(target_folder) 

	import pandas as pd
	df = pd.read_csv(csv)

	import shutil 
	IDs = df['MRI.ID'].values.tolist()


	for ID in IDs:
		if os.path.isfile(source_folder + ID + '.nii.gz'):
	    		shutil.copyfile(source_folder + ID + '.nii.gz', target_folder + ID + '.nii.gz')

	os.chdir(target_folder)

	from subprocess import call
	command = 'tbss_1_preproc *.nii.gz'
	call(command,shell=True)

	os.chdir(target_folder)
	from subprocess import call
	command = 'tbss_2_reg -T'
	call(command,shell=True)
	"""
	command = 'tbss_3_postreg -S'
	call(command,shell=True)

	command = 'tbss_4_prestats 0.2'
	call(command,shell=True)
	"""
if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())
