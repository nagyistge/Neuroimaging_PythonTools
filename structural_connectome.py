def FA_connectome_CSDeterministic(subject_list,base_directory,out_directory):

	#==============================================================
	# Loading required packages
	import nipype.interfaces.io as nio
	import nipype.pipeline.engine as pe
	import nipype.interfaces.utility as util
	import nipype.interfaces.fsl as fsl
	import nipype.interfaces.dipy as dipy
	import nipype.interfaces.mrtrix as mrt
	from own_nipype import DipyDenoise as denoise
	from own_nipype import trk_Coreg as trkcoreg
	from own_nipype import TXT2PCK as txt2pck
	from own_nipype import FAconnectome as connectome
	from own_nipype import Extractb0 as extract_b0
	import nipype.interfaces.cmtk as cmtk
	import nipype.interfaces.diffusion_toolkit as dtk
	import nipype.algorithms.misc as misc

	from nipype import SelectFiles
	import os
	registration_reference = os.environ['FSLDIR'] + '/data/standard/MNI152_T1_1mm_brain.nii.gz'
	nodes = list()

	#====================================
	# Defining the nodes for the workflow

	# Utility nodes
	gunzip = pe.Node(interface=misc.Gunzip(), name='gunzip')
	gunzip2 = pe.Node(interface=misc.Gunzip(), name='gunzip2')
	fsl2mrtrix = pe.Node(interface=mrt.FSL2MRTrix(invert_x=True),name='fsl2mrtrix')

	# Getting the subject ID
	infosource  = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),name='infosource')
	infosource.iterables = ('subject_id', subject_list)

	# Getting the relevant diffusion-weighted data
	templates = dict(dwi='{subject_id}/dwi/{subject_id}_dwi.nii.gz',
		bvec='{subject_id}/dwi/{subject_id}_dwi.bvec',
		bval='{subject_id}/dwi/{subject_id}_dwi.bval')

	selectfiles = pe.Node(SelectFiles(templates),
	                   name='selectfiles')
	selectfiles.inputs.base_directory = os.path.abspath(base_directory)

	# Denoising
	denoise = pe.Node(interface=denoise(), name='denoise')

	# Eddy-current and motion correction
	eddycorrect = pe.Node(interface=fsl.epi.EddyCorrect(), name='eddycorrect')
	eddycorrect.inputs.ref_num = 0

	# Upsampling
	resample = pe.Node(interface=dipy.Resample(interp=3,vox_size=(1.,1.,1.)), name='resample')

	# Extract b0 image
	extract_b0 = pe.Node(interface=extract_b0(),name='extract_b0')

	# Fitting the diffusion tensor model
	dwi2tensor = pe.Node(interface=mrt.DWI2Tensor(), name='dwi2tensor')
	tensor2vector = pe.Node(interface=mrt.Tensor2Vector(), name='tensor2vector')
	tensor2adc = pe.Node(interface=mrt.Tensor2ApparentDiffusion(), name='tensor2adc')
	tensor2fa = pe.Node(interface=mrt.Tensor2FractionalAnisotropy(), name='tensor2fa')

	# Create a brain mask
	bet = pe.Node(interface=fsl.BET(frac=0.3,robust=False,mask=True),name='bet')

	# Eroding the brain mask
	erode_mask_firstpass = pe.Node(interface=mrt.Erode(), name='erode_mask_firstpass')
	erode_mask_secondpass = pe.Node(interface=mrt.Erode(), name='erode_mask_secondpass')
	MRmultiply = pe.Node(interface=mrt.MRMultiply(), name='MRmultiply')
	MRmult_merge = pe.Node(interface=util.Merge(2), name='MRmultiply_merge')
	threshold_FA = pe.Node(interface=mrt.Threshold(absolute_threshold_value = 0.7), name='threshold_FA')

	# White matter mask
	gen_WM_mask = pe.Node(interface=mrt.GenerateWhiteMatterMask(), name='gen_WM_mask')
	threshold_wmmask = pe.Node(interface=mrt.Threshold(absolute_threshold_value = 0.4), name='threshold_wmmask')

	# CSD probabilistic tractography 
	estimateresponse = pe.Node(interface=mrt.EstimateResponseForSH(maximum_harmonic_order = 8), name='estimateresponse')
	csdeconv = pe.Node(interface=mrt.ConstrainedSphericalDeconvolution(maximum_harmonic_order = 8), name='csdeconv')

	# Tracking 
	detCSDstreamtrack = pe.Node(interface=mrt.ProbabilisticSphericallyDeconvolutedStreamlineTrack(), name='detCSDstreamtrack')
	detCSDstreamtrack.inputs.inputmodel = 'SD_STREAM'
	tck2trk = pe.Node(interface=mrt.MRTrix2TrackVis(), name='tck2trk')

	# smoothing the tracts 
	smooth = pe.Node(interface=dtk.SplineFilter(step_length=0.5), name='smooth')

	# Co-registration with MNI space
	mrconvert = pe.Node(mrt.MRConvert(extension='nii'), name='mrconvert')
	flt = pe.Node(interface=fsl.FLIRT(reference=registration_reference, dof=12, cost_func='corratio'), name='flt')

	# Moving tracts to common space
	trkcoreg = pe.Node(interface=trkcoreg(reference=registration_reference),name='trkcoreg_det')

	# calcuating the connectome matrix 
	calc_matrix = pe.Node(interface=connectome(ROI_file='/home/jb07/Desktop/aal.nii.gz'),name='calc_matrix_det')

	# Converting the adjacency matrix from txt to pck format
	txt2pck = pe.Node(interface=txt2pck(), name='txt2pck_det')

	# Calculate graph theory measures with NetworkX and CMTK
	nxmetrics = pe.Node(interface=cmtk.NetworkXMetrics(treat_as_weighted_graph = True), name='nxmetrics_det')

	#====================================
	# Setting up the workflow
	fa_connectome = pe.Workflow(name='FA_connectome')

	# Reading in files
	fa_connectome.connect(infosource, 'subject_id', selectfiles, 'subject_id')

	# Denoising
	fa_connectome.connect(selectfiles, 'dwi', denoise, 'in_file')

	# Eddy current and motion correction
	fa_connectome.connect(denoise, 'out_file',eddycorrect, 'in_file')
	fa_connectome.connect(eddycorrect, 'eddy_corrected', resample, 'in_file')
	fa_connectome.connect(resample, 'out_file', extract_b0, 'in_file')
	fa_connectome.connect(resample, 'out_file', gunzip,'in_file')

	# Brain extraction
	fa_connectome.connect(extract_b0, 'out_file', bet, 'in_file')

	# Creating tensor maps
	fa_connectome.connect(selectfiles,'bval',fsl2mrtrix,'bval_file')
	fa_connectome.connect(selectfiles,'bvec',fsl2mrtrix,'bvec_file')
	fa_connectome.connect(gunzip,'out_file',dwi2tensor,'in_file')
	fa_connectome.connect(fsl2mrtrix,'encoding_file',dwi2tensor,'encoding_file')
	fa_connectome.connect(dwi2tensor,'tensor',tensor2vector,'in_file')
	fa_connectome.connect(dwi2tensor,'tensor',tensor2adc,'in_file')
	fa_connectome.connect(dwi2tensor,'tensor',tensor2fa,'in_file')
	fa_connectome.connect(tensor2fa,'FA', MRmult_merge, 'in1')

	# Thresholding to create a mask of single fibre voxels
	fa_connectome.connect(gunzip2, 'out_file', erode_mask_firstpass, 'in_file')
	fa_connectome.connect(erode_mask_firstpass, 'out_file', erode_mask_secondpass, 'in_file')
	fa_connectome.connect(erode_mask_secondpass,'out_file', MRmult_merge, 'in2')
	fa_connectome.connect(MRmult_merge, 'out', MRmultiply,  'in_files')
	fa_connectome.connect(MRmultiply, 'out_file', threshold_FA, 'in_file')

	# Create seed mask
	fa_connectome.connect(gunzip, 'out_file', gen_WM_mask, 'in_file')
	fa_connectome.connect(bet, 'mask_file', gunzip2, 'in_file')
	fa_connectome.connect(gunzip2, 'out_file', gen_WM_mask, 'binary_mask')
	fa_connectome.connect(fsl2mrtrix, 'encoding_file', gen_WM_mask, 'encoding_file')
	fa_connectome.connect(gen_WM_mask, 'WMprobabilitymap', threshold_wmmask, 'in_file')

	# Estimate response
	fa_connectome.connect(gunzip, 'out_file', estimateresponse, 'in_file')
	fa_connectome.connect(fsl2mrtrix, 'encoding_file', estimateresponse, 'encoding_file')
	fa_connectome.connect(threshold_FA, 'out_file', estimateresponse, 'mask_image')

	# CSD calculation
	fa_connectome.connect(gunzip, 'out_file', csdeconv, 'in_file')
	fa_connectome.connect(gen_WM_mask, 'WMprobabilitymap', csdeconv, 'mask_image')
	fa_connectome.connect(estimateresponse, 'response', csdeconv, 'response_file')
	fa_connectome.connect(fsl2mrtrix, 'encoding_file', csdeconv, 'encoding_file')

	# Running the tractography
	fa_connectome.connect(threshold_wmmask, "out_file", detCSDstreamtrack, "seed_file")
	fa_connectome.connect(csdeconv, "spherical_harmonics_image", detCSDstreamtrack, "in_file")
	fa_connectome.connect(gunzip, "out_file", tck2trk, "image_file")
	fa_connectome.connect(detCSDstreamtrack, "tracked", tck2trk, "in_file")

	# Smoothing the trackfile
	fa_connectome.connect(tck2trk, 'out_file',smooth,'track_file')

	# Co-registering FA with FMRIB58_FA_1mm standard space 
	fa_connectome.connect(MRmultiply,'out_file',mrconvert,'in_file')
	fa_connectome.connect(mrconvert,'converted',flt,'in_file')
	fa_connectome.connect(smooth,'smoothed_track_file',trkcoreg,'in_file')
	fa_connectome.connect(mrconvert,'converted',trkcoreg,'FA_file')
	fa_connectome.connect(flt,'out_matrix_file',trkcoreg,'transfomation_matrix')

	# Calculating the FA connectome
	fa_connectome.connect(trkcoreg,'transformed_track_file',calc_matrix,'trackfile')
	fa_connectome.connect(flt,'out_file',calc_matrix,'FA_file')

	# Calculating graph measures 
	fa_connectome.connect(calc_matrix,'out_file',txt2pck,'in_file')
	fa_connectome.connect(txt2pck,'out_file',nxmetrics,'in_file')

	#====================================
	# Running the workflow
	fa_connectome.base_dir = os.path.abspath(out_directory)
	fa_connectome.write_graph()
	fa_connectome.run('PBSGraph')