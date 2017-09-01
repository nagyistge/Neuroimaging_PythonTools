def DiPy_CSD(participant):
	import os 
	import sys
	sys.path.append('/home/jb07/python_modules/')
	os.chdir('/imaging/jb07/CALM/DWI/DiPy_CSD/')
	raw_folder = '/imaging/jb07/CALM/raw_data/DTI/'
	dwi_folder = '/imaging/jb07/CALM/DWI/upsampled/'
	FA_folder = '/imaging/jb07/CALM/DWI/FA_masked/'


	# Loading the data
	import nibabel as nib
	import numpy as np
	img = nib.load(dwi_folder + participant + '.nii')
	data = img.get_data()
	affine = np.round(img.affine)

	FA_img = nib.load(FA_folder + participant + '.nii')
	FA = FA_img.get_data()
	white_matter = FA >= 0.2

	import numpy as np
	bvals = np.loadtxt(raw_folder + participant + '.bvals')
	bvecs = np.loadtxt(raw_folder + participant + '.bvecs').T
	bvecs = np.vstack([bvecs[0,:],bvecs[1,:],-1*bvecs[2,:]])

	from dipy.core.gradients import gradient_table
	gtab = gradient_table(bvals.T,bvecs)

	#============================================================

	from dipy.reconst.shm import CsaOdfModel
	from dipy.data import default_sphere
	from dipy.direction import peaks_from_model
	csa_model = CsaOdfModel(gtab, sh_order=8)
	csa_peaks = peaks_from_model(csa_model, data, default_sphere,
	                             relative_peak_threshold=.8,
	                             min_separation_angle=45,
	                             mask=white_matter)

	from dipy.tracking.local import ThresholdTissueClassifier
	classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)

	from dipy.tracking import utils
	seeds = utils.seeds_from_mask(white_matter, density=[1, 1, 1], affine=affine)

	from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
	                                   auto_response)

	response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.6)
	csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
	csd_fit = csd_model.fit(data, mask=white_matter)

	from dipy.direction import ProbabilisticDirectionGetter
	from dipy.direction import DeterministicMaximumDirectionGetter
	from dipy.data import default_sphere

	det_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
	                                                             max_angle=45.,
	                                                             sphere=default_sphere)
	from dipy.tracking.local import LocalTracking
	streamlines = LocalTracking(det_dg, classifier, seeds, affine,
	                            step_size=.5)

	# Compute streamlines and store as a list.
	streamlines = list(streamlines)

	# Make a trackvis header so we can save streamlines
	import nibabel as nib
	voxel_size = FA_img.get_header().get_zooms()
	trackvis_header = nib.trackvis.empty_header()
	trackvis_header['voxel_size'] = voxel_size
	trackvis_header['dim'] = FA.shape
	trackvis_header['voxel_order'] = "RAS"

	# Move streamlines to "trackvis space"
	trackvis_point_space = utils.affine_for_trackvis(voxel_size)
	streamlines_trk = utils.move_streamlines(streamlines,
	                                   trackvis_point_space, input_space=affine)
	streamlines_trk = list(streamlines_trk)

	# Save streamlines
	for_save = [(sl, None, None) for sl in streamlines_trk]
	nib.trackvis.write("CSD_tractography_own.trk", for_save, trackvis_header)
