#! /usr/bin/env python
import sys
import optparse

def main():
	p = optparse.OptionParser()
	p.add_option('--participant', '-p')
	p.add_option('--stage', '-s')
	options, arguments = p.parse_args()
	participant = options.participant
	stage = options.stage

	sys.path.append('/home/jb07/python_modules/')

	import DWI_preprocess
	import DWI_tracking
	import os

	try: 
		dwi_image =  '/imaging/jb07/CALM/DWI/upsampled/' + participant + '.nii'
		encoding_file = '/imaging/jb07/CALM/raw_data/DTI/' + participant + '_encoding.txt'
		brain_mask = '/imaging/jb07/CALM/DWI/brain_mask/' + participant + '_brain_mask.nii'
		masked_FA = '/imaging/jb07/CALM/DWI/FA_masked/' + participant + '.nii'
		CSD_image = '/imaging/jb07/CALM/DWI/MRTrix/' + participant + '_CSD8.nii'

		if stage == 'all':			
			print 'all'
			os.chdir('/imaging/jb07/CALM/DWI/MRTrix/')
			DWI_preprocess.preprocess_dwi(participant + '.nii.gz')
			DWI_tracking.fit_CSD_model(participant,dwi_image,encoding_file,brain_mask,masked_FA)
			DWI_tracking.track_whole_brain(masked_FA,CSD_image,brain_mask,participant)
			DWI_tracking.additional_tractography_files(argument + '.nii.gz')

		if str(stage) == '1':
			os.chdir('/imaging/jb07/CALM/DWI/MRTrix/')
			DWI_preprocess.preprocess_dwi(participant + '.nii.gz')

		if str(stage) == '2':
			os.chdir('/imaging/jb07/CALM/DWI/MRTrix/')
			DWI_tracking.fit_CSD_model(participant,dwi_image,encoding_file,brain_mask,masked_FA)
			DWI_tracking.track_whole_brain(masked_FA,CSD_image,brain_mask,participant)
			DWI_tracking.additional_tractography_files(participant + '.nii.gz')

		if str(stage) == '3':
			os.chdir('/imaging/jb07/CALM/DWI/MRTrix/')
			DWI_tracking.track_whole_brain(masked_FA,CSD_image,brain_mask,participant)
			DWI_tracking.additional_tractography_files(participant + '.nii.gz')

		if str(stage) == '4':
			DWI_tracking.additional_tractography_files(participant + '.nii.gz')

	except:
		print 'ERROR'

if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())
