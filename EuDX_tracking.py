def EuDX_tracking(participant,outfile):
	import sys
	sys.path.append('/home/jb07/python_module/')

	# Loading the diffusion data and brain mask
	import nibabel as nib
	img = nib.load('/imaging/jb07/CALM/DWI/upsampled/' + participant + '.nii')
	data = img.get_data()
	mask_img = nib.load('/imaging/jb07/CALM/DWI/brain_mask/' + participant + '_brain_mask.nii')
	mask = mask_img.get_data()

	# Reading the gradient table
	from dipy.reconst.csdeconv import auto_response
	from dipy.core.gradients import gradient_table
	gtab = gradient_table('/imaging/jb07/CALM/raw_data/DTI/' + participant + '.bvals', '/imaging/jb07/CALM/raw_data/DTI/' + participant + '.bvecs')
	response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

	# Fitting the CSD model
	from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
	csd_model = ConstrainedSphericalDeconvModel(gtab, response)

	# Getting the peaks
	from dipy.direction import peaks_from_model
	from dipy.data import get_sphere
	sphere = get_sphere('symmetric724')

	csapeaks = peaks_from_model(model=csd_model,
	                                      data=data,
	                                      sphere=sphere,
	                                      relative_peak_threshold=.5,
	                                      min_separation_angle=25,
	                                      mask=mask,
	                                      return_sh=True,
	                                      return_odf=False,
	                                      normalize_peaks=True,
	                                      npeaks=5,
	                                      parallel=True)

	# Tracking 
	from dipy.tracking.eudx import EuDX

	eu = EuDX(csapeaks.gfa,
	          csapeaks.peak_indices[..., 0],
	          seeds=10000,
	          odf_vertices=sphere.vertices,
	          a_low=0.2)

	csa_streamlines = [streamline for streamline in eu]

	# Saving the trackfile
	import nibabel as nib
	hdr = nib.trackvis.empty_header()
	hdr['voxel_size'] = (1., 1., 1.)
	hdr['voxel_order'] = 'LAS'
	hdr['dim'] = csapeaks.gfa.shape[:3]

	csa_streamlines_trk = ((sl, None, None) for sl in csa_streamlines)
	nib.trackvis.write(outfile, csa_streamlines_trk, hdr, points_space='voxel')

EuDX_tracking('CBU150085','/home/jb07/Desktop/test.trk')