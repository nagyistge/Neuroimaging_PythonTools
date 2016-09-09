def dwi_preproc(subject_list,base_directory,out_directory):
	"""
	This function implements the dwi preprocessing workflow. The function takes a list of subject IDs and their parent directory - the data is expected to be stored according to the Brain Imaging Data Structure (BIDS).
	It then performs the preprocessing steps: denoising with non-local means (http://nipy.org/dipy/examples_built/denoise_nlmeans.html), FSL eddy_correct to correct for eddy current and participant motion, resampling to 1mm isotropic resolution with trilinear interpolation, extraction of the first b0 volume, brain extraction with FSL bet, and fitting of the diffusion tensor model with FSL dtifit

	inputs:
	subject list: python list of string with the subject IDs
	base_directory: directory in which the raw data is stored (diffusion weighted volume, bval and bvecs file)
	out_directory: directory where is output will be stored

	written by Joe Bathelt,PhD
	MRC Cognition & Brain Sciences Unit
	joe.bathelt@mrc-cbu.cam.ac.uk
	"""
	#==============================================================
	# Loading required packages
	import nipype.interfaces.io as nio
	import nipype.pipeline.engine as pe
	import nipype.interfaces.utility as util
	import nipype.interfaces.fsl as fsl
	import nipype.interfaces.dipy as dipy
	from own_nipype import DipyDenoise as denoise
	import nipype.interfaces.diffusion_toolkit as dtk

	from nipype import SelectFiles
	import os
	nodes = list()

	#====================================
	# Defining the nodes for the workflow

	# Getting the subject ID
	infosource  = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),name='infosource')
	infosource.iterables = ('subject_id', subject_list)

	# Getting the relevant diffusion-weighted data
	templates = dict(dwi='{subject_id}/dwi/{subject_id}_dwi.nii.gz',
		bvec='{subject_id}/dwi/{subject_id}_dwi.bvec',
		bval='{subject_id}/dwi/{subject_id}_dwi.bval')

	selectfiles = pe.Node(SelectFiles(templates),
	                   name="selectfiles")
	selectfiles.inputs.base_directory = os.path.abspath(base_directory)
	nodes.append(selectfiles)

	# Denoising
	denoise = pe.Node(interface=denoise(), name='denoise')
	nodes.append(denoise)

	# Eddy-current and motion correction
	eddycorrect = pe.Node(interface=fsl.epi.EddyCorrect(), name='eddycorrect')
	eddycorrect.inputs.ref_num = 0
	nodes.append(eddycorrect)

	# Upsampling
	resample = pe.Node(interface=dipy.Resample(interp=3,vox_size=(1.,1.,1.)), name='resample')
	nodes.append(resample)

	# Extract b0 image
	fslroi = pe.Node(interface=fsl.ExtractROI(),name='extract_b0')
	fslroi.inputs.t_min=0
	fslroi.inputs.t_size=1
	nodes.append(fslroi)

	# Create a brain mask
	bet = pe.Node(interface=fsl.BET(frac=0.3,robust=False,mask=True),name='bet')
	nodes.append(bet)

	# Fitting the diffusion tensor model
	dtifit = pe.Node(interface=fsl.DTIFit(),name='dtifit')
	nodes.append(dtifit)

	#====================================
	# Setting up the workflow
	dwi_preproc = pe.Workflow(name='dwi_preproc')
	dwi_preproc.add_nodes(nodes)

	dwi_preproc.connect(infosource, 'subject_id', selectfiles, 'subject_id')
	dwi_preproc.connect(selectfiles, 'dwi', denoise, 'in_file')
	dwi_preproc.connect(denoise, 'out_file', eddycorrect, 'in_file')
	dwi_preproc.connect(eddycorrect, 'eddy_corrected', resample, 'in_file')
	dwi_preproc.connect(resample, 'out_file', fslroi, 'in_file')
	dwi_preproc.connect(fslroi, 'roi_file', bet, 'in_file')

	dwi_preproc.connect(infosource, 'subject_id',dtifit,'base_name')
	dwi_preproc.connect(resample, 'out_file',dtifit,'dwi')
	dwi_preproc.connect(selectfiles, 'bvec',dtifit,'bvecs')
	dwi_preproc.connect(selectfiles, 'bval',dtifit,'bvals')
	dwi_preproc.connect(bet,'mask_file',dtifit,'mask')

	#====================================
	# Running the workflow
	dwi_preproc.base_dir = os.path.abspath(out_directory)
	dwi_preproc.write_graph()
	dwi_preproc.run()


def dwi_preproc_minimal(subject_list,base_directory,out_directory):
	"""
	This function implements the dwi preprocessing workflow. The function takes a list of subject IDs and their parent directory - the data is expected to be stored according to the Brain Imaging Data Structure (BIDS).
	It then performs the preprocessing steps: denoising with non-local means (http://nipy.org/dipy/examples_built/denoise_nlmeans.html), FSL eddy_correct to correct for eddy current and participant motion, resampling to 1mm isotropic resolution with trilinear interpolation, extraction of the first b0 volume, brain extraction with FSL bet, and fitting of the diffusion tensor model with FSL dtifit

	inputs:
	subject list: python list of string with the subject IDs
	base_directory: directory in which the raw data is stored (diffusion weighted volume, bval and bvecs file)
	out_directory: directory where is output will be stored

	written by Joe Bathelt,PhD
	MRC Cognition & Brain Sciences Unit
	joe.bathelt@mrc-cbu.cam.ac.uk
	"""
	#==============================================================
	# Loading required packages
	import nipype.interfaces.io as nio
	import nipype.pipeline.engine as pe
	import nipype.interfaces.utility as util
	import nipype.interfaces.fsl as fsl
	import nipype.interfaces.dipy as dipy
	from own_nipype import DipyDenoise as denoise

	from nipype import SelectFiles
	import os
	nodes = list()

	#====================================
	# Defining the nodes for the workflow

	# Getting the subject ID
	infosource  = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),name='infosource')
	infosource.iterables = ('subject_id', subject_list)

	# Getting the relevant diffusion-weighted data
	templates = dict(dwi='{subject_id}/dwi/{subject_id}_dwi.nii.gz',
		bvec='{subject_id}/dwi/{subject_id}_dwi.bvec',
		bval='{subject_id}/dwi/{subject_id}_dwi.bval')

	selectfiles = pe.Node(SelectFiles(templates),
	                   name="selectfiles")
	selectfiles.inputs.base_directory = os.path.abspath(base_directory)
	nodes.append(selectfiles)

	# Denoising
	denoise = pe.Node(interface=denoise(), name='denoise')
	nodes.append(denoise)

	# Eddy-current and motion correction
	eddycorrect = pe.Node(interface=fsl.epi.EddyCorrect(), name='eddycorrect')
	eddycorrect.inputs.ref_num = 0
	nodes.append(eddycorrect)

	# Extract b0 image
	fslroi = pe.Node(interface=fsl.ExtractROI(),name='extract_b0')
	fslroi.inputs.t_min=0
	fslroi.inputs.t_size=1
	nodes.append(fslroi)

	# Create a brain mask
	bet = pe.Node(interface=fsl.BET(frac=0.3,robust=False,mask=True),name='bet')
	nodes.append(bet)

	# Erode brain mask
	erode = pe.Node(interface=fsl.maths.ErodeImage(),name='erode')
	nodes.append(erode)

	# Fitting the diffusion tensor model
	dtifit = pe.Node(interface=fsl.DTIFit(),name='dtifit')
	nodes.append(dtifit)

	#====================================
	# Setting up the workflow
	dwi_preproc_minimal = pe.Workflow(name='dwi_preproc_minimal')
	dwi_preproc_minimal.add_nodes(nodes)

	dwi_preproc_minimal.connect(infosource, 'subject_id', selectfiles, 'subject_id')
	dwi_preproc_minimal.connect(selectfiles, 'dwi', eddycorrect, 'in_file')
	dwi_preproc_minimal.connect(eddycorrect, 'eddy_corrected', fslroi, 'in_file')
	dwi_preproc_minimal.connect(fslroi, 'roi_file', bet, 'in_file')

	dwi_preproc_minimal.connect(infosource, 'subject_id',dtifit,'base_name')
	dwi_preproc_minimal.connect(eddycorrect, 'eddy_corrected', dtifit,'dwi')
	dwi_preproc_minimal.connect(selectfiles, 'bvec',dtifit,'bvecs')
	dwi_preproc_minimal.connect(selectfiles, 'bval',dtifit,'bvals')
	dwi_preproc_minimal.connect(bet,'mask_file',dtifit,'mask')

	dwi_preproc_minimal.connect(bet,'mask_file',erode,'in_file')

	#====================================
	# Running the workflow
	dwi_preproc_minimal.base_dir = os.path.abspath(out_directory)
	dwi_preproc_minimal.write_graph()
	dwi_preproc_minimal.run()

def dwi_preproc_restore(subject_list,base_directory,out_directory):
	"""
	This function implements the dwi preprocessing workflow. The function takes a list of subject IDs and their parent directory - the data is expected to be stored according to the Brain Imaging Data Structure (BIDS).
	It then performs the preprocessing steps: denoising with non-local means (http://nipy.org/dipy/examples_built/denoise_nlmeans.html), FSL eddy_correct to correct for eddy current and participant motion, resampling to 1mm isotropic resolution with trilinear interpolation, extraction of the first b0 volume, brain extraction with FSL bet, and fitting of the diffusion tensor model with FSL dtifit

	inputs:
	subject list: python list of string with the subject IDs
	base_directory: directory in which the raw data is stored (diffusion weighted volume, bval and bvecs file)
	out_directory: directory where is output will be stored

	written by Joe Bathelt,PhD
	MRC Cognition & Brain Sciences Unit
	joe.bathelt@mrc-cbu.cam.ac.uk
	"""
	#==============================================================
	# Loading required packages
	import nipype.interfaces.io as nio
	import nipype.pipeline.engine as pe
	import nipype.interfaces.utility as util
	import nipype.interfaces.fsl as fsl
	import nipype.interfaces.dipy as dipy
	from own_nipype import DipyDenoise as denoise
	from own_nipype import DipyRestore as restore

	from nipype import SelectFiles
	import os
	nodes = list()

	#====================================
	# Defining the nodes for the workflow

	# Getting the subject ID
	infosource  = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),name='infosource')
	infosource.iterables = ('subject_id', subject_list)

	# Getting the relevant diffusion-weighted data
	templates = dict(dwi='{subject_id}/dwi/{subject_id}_dwi.nii.gz',
		bvec='{subject_id}/dwi/{subject_id}_dwi.bvec',
		bval='{subject_id}/dwi/{subject_id}_dwi.bval')

	selectfiles = pe.Node(SelectFiles(templates),
	                   name="selectfiles")
	selectfiles.inputs.base_directory = os.path.abspath(base_directory)
	nodes.append(selectfiles)

	# Denoising
	denoise = pe.Node(interface=denoise(), name='denoise')
	nodes.append(denoise)

	# Eddy-current and motion correction
	eddycorrect = pe.Node(interface=fsl.epi.EddyCorrect(), name='eddycorrect')
	eddycorrect.inputs.ref_num = 0
	nodes.append(eddycorrect)

	# Extract b0 image
	fslroi = pe.Node(interface=fsl.ExtractROI(),name='extract_b0')
	fslroi.inputs.t_min=0
	fslroi.inputs.t_size=1
	nodes.append(fslroi)

	# Create a brain mask
	bet = pe.Node(interface=fsl.BET(frac=0.3,robust=False,mask=True),name='bet')
	nodes.append(bet)

	# Erode brain mask
	erode = pe.Node(interface=fsl.maths.ErodeImage(),name='erode')
	nodes.append(erode)

	# Fitting the diffusion tensor model
	restore = pe.Node(interface=restore(),name='restore')
	nodes.append(restore)

	#====================================
	# Setting up the workflow
	dwi_preproc_restore = pe.Workflow(name='dwi_preproc_restore')
	dwi_preproc_restore.add_nodes(nodes)

	dwi_preproc_restore.connect(infosource, 'subject_id', selectfiles, 'subject_id')
	dwi_preproc_restore.connect(selectfiles, 'dwi', denoise, 'in_file')
	dwi_preproc_restore.connect(denoise, 'out_file',eddycorrect, 'in_file')
	dwi_preproc_restore.connect(eddycorrect, 'eddy_corrected', fslroi, 'in_file')
	dwi_preproc_restore.connect(fslroi, 'roi_file', bet, 'in_file')

	dwi_preproc_restore.connect(eddycorrect, 'eddy_corrected', restore,'in_file')
	dwi_preproc_restore.connect(selectfiles, 'bvec',restore,'bvec')
	dwi_preproc_restore.connect(selectfiles, 'bval',restore,'bval')

	dwi_preproc_restore.connect(bet,'mask_file',erode,'in_file')

	#====================================
	# Running the workflow
	dwi_preproc_restore.base_dir = os.path.abspath(out_directory)
	dwi_preproc_restore.write_graph()
	dwi_preproc_restore.run()



def CSD_deterministic_tractography(subject_list,base_directory,out_directory):
	#==============================================================
	# Loading required packages
	import nipype.interfaces.io as nio
	import nipype.pipeline.engine as pe
	import nipype.interfaces.utility as util
	import nipype.interfaces.fsl as fsl
	import nipype.interfaces.dipy as dipy
	from own_nipype import CSDdet as csdet
	from own_nipype import DipyDenoise as denoise
	import nipype.interfaces.diffusion_toolkit as dtk

	from nipype import SelectFiles
	import os
	nodes = list()

	#====================================
	# Defining the nodes for the workflow

	# Getting the subject ID
	infosource  = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),name='infosource')
	infosource.iterables = ('subject_id', subject_list)

	# Getting the relevant diffusion-weighted data
	templates = dict(dwi='{subject_id}/dwi/{subject_id}_dwi.nii.gz',
		bvec='{subject_id}/dwi/{subject_id}_dwi.bvec',
		bval='{subject_id}/dwi/{subject_id}_dwi.bval')

	selectfiles = pe.Node(SelectFiles(templates),
	                   name="selectfiles")
	selectfiles.inputs.base_directory = os.path.abspath(base_directory)
	nodes.append(selectfiles)

	# Denoising
	denoise = pe.Node(interface=denoise(), name='denoise')
	nodes.append(denoise)

	# Eddy-current and motion correction
	eddycorrect = pe.Node(interface=fsl.epi.EddyCorrect(), name='eddycorrect')
	eddycorrect.inputs.ref_num = 0
	nodes.append(eddycorrect)

	# Upsampling
	resample = pe.Node(interface=dipy.Resample(interp=3,vox_size=(1.,1.,1.)), name='resample')
	nodes.append(resample)

	# Extract b0 image
	fslroi = pe.Node(interface=fsl.ExtractROI(),name='extract_b0')
	fslroi.inputs.t_min=0
	fslroi.inputs.t_size=1
	nodes.append(fslroi)

	# Create a brain mask
	bet = pe.Node(interface=fsl.BET(frac=0.3,robust=False,mask=True),name='bet')
	nodes.append(bet)

	# Erode brain mask
	erode = pe.Node(interface=fsl.maths.ErodeImage(),name='erode')
	nodes.append(erode)

	# Fitting the diffusion tensor model
	dtifit = pe.Node(interface=fsl.DTIFit(),name='dtifit')
	nodes.append(dtifit)

	# CSD deterministic tractography
	csdet = pe.Node(interface=csdet(),name='csdet')
	nodes.append(csdet)

	# smoothing the tracts
	smooth = pe.Node(interface=dtk.SplineFilter(step_length=0.5), name='smooth')
	nodes.append(smooth)

	#====================================
	# Setting up the workflow
	csd_det = pe.Workflow(name='dwi_preproc_minimal')
	csd_det.add_nodes(nodes)

	csd_det.connect(infosource, 'subject_id', selectfiles, 'subject_id')
	csd_det.connect(selectfiles, 'dwi', denoise, 'in_file')
	csd_det.connect(denoise, 'out_file',eddycorrect, 'in_file')
	csd_det.connect(eddycorrect, 'eddy_corrected', resample, 'in_file')
	csd_det.connect(resample, 'out_file', fslroi, 'in_file')
	csd_det.connect(fslroi, 'roi_file', bet, 'in_file')

	csd_det.connect(infosource, 'subject_id',dtifit,'base_name')
	csd_det.connect(resample, 'out_file', dtifit,'dwi')
	csd_det.connect(selectfiles, 'bvec',dtifit,'bvecs')
	csd_det.connect(selectfiles, 'bval',dtifit,'bvals')
	csd_det.connect(bet,'mask_file',dtifit,'mask')

	csd_det.connect(bet,'mask_file',erode,'in_file')

	csd_det.connect(erode,'out_file',csdet,'brain_mask')
	csd_det.connect(dtifit,'FA',csdet,'FA_file')
	csd_det.connect(selectfiles,'bval',csdet,'bval')
	csd_det.connect(selectfiles,'bvec',csdet,'bvec')
	csd_det.connect(eddycorrect, 'eddy_corrected',csdet,'in_file')

	csd_det.connect(csdet, 'out_file',smooth,'track_file')

	#====================================
	# Running the workflow
	csd_det.base_dir = os.path.abspath(out_directory)
	csd_det.write_graph()
	csd_det.run()

def CSD_probablistic_tractography(subject_list,base_directory,out_directory):
	#==============================================================
	# Loading required packages
	import nipype.interfaces.io as nio
	import nipype.pipeline.engine as pe
	import nipype.interfaces.utility as util
	import nipype.interfaces.fsl as fsl
	import nipype.interfaces.dipy as dipy
	from own_nipype import CSDprob as csdprob
	from own_nipype import DipyDenoise as denoise
	import nipype.interfaces.diffusion_toolkit as dtk

	from nipype import SelectFiles
	import os
	nodes = list()

	#====================================
	# Defining the nodes for the workflow

	# Getting the subject ID
	infosource  = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),name='infosource')
	infosource.iterables = ('subject_id', subject_list)

	# Getting the relevant diffusion-weighted data
	templates = dict(dwi='{subject_id}/dwi/{subject_id}_dwi.nii.gz',
		bvec='{subject_id}/dwi/{subject_id}_dwi.bvec',
		bval='{subject_id}/dwi/{subject_id}_dwi.bval')

	selectfiles = pe.Node(SelectFiles(templates),
	                   name="selectfiles")
	selectfiles.inputs.base_directory = os.path.abspath(base_directory)
	nodes.append(selectfiles)

	# Eddy-current and motion correction
	eddycorrect = pe.Node(interface=fsl.epi.EddyCorrect(), name='eddycorrect')
	eddycorrect.inputs.ref_num = 0
	nodes.append(eddycorrect)

	# Extract b0 image
	fslroi = pe.Node(interface=fsl.ExtractROI(),name='extract_b0')
	fslroi.inputs.t_min=0
	fslroi.inputs.t_size=1
	nodes.append(fslroi)

	# Create a brain mask
	bet = pe.Node(interface=fsl.BET(frac=0.3,robust=False,mask=True),name='bet')
	nodes.append(bet)

	# Erode brain mask
	erode = pe.Node(interface=fsl.maths.ErodeImage(),name='erode')
	nodes.append(erode)

	# Fitting the diffusion tensor model
	dtifit = pe.Node(interface=fsl.DTIFit(),name='dtifit')
	nodes.append(dtifit)

	# CSD probabilistic tractography
	csdprob = pe.Node(interface=csdprob(),name='csdprob')
	nodes.append(csdprob)

	# smoothing the tracts
	smooth = pe.Node(interface=dtk.SplineFilter(step_length=0.5), name='smooth')
	nodes.append(smooth)

	#====================================
	# Setting up the workflow
	csd_prob = pe.Workflow(name='CSD_probablistic_tractography')
	csd_prob.add_nodes(nodes)

	csd_prob.connect(infosource, 'subject_id', selectfiles, 'subject_id')
	csd_prob.connect(selectfiles, 'dwi',eddycorrect, 'in_file')
	csd_prob.connect(eddycorrect, 'eddy_corrected', fslroi, 'in_file')
	csd_prob.connect(fslroi, 'roi_file', bet, 'in_file')

	csd_prob.connect(infosource, 'subject_id',dtifit,'base_name')
	csd_prob.connect(eddycorrect, 'eddy_corrected', dtifit,'dwi')
	csd_prob.connect(selectfiles, 'bvec',dtifit,'bvecs')
	csd_prob.connect(selectfiles, 'bval',dtifit,'bvals')
	csd_prob.connect(bet,'mask_file',dtifit,'mask')

	csd_prob.connect(bet,'mask_file',erode,'in_file')

	csd_prob.connect(erode,'out_file',csdprob,'brain_mask')
	csd_prob.connect(dtifit,'FA',csdprob,'FA_file')
	csd_prob.connect(selectfiles,'bval',csdprob,'bval')
	csd_prob.connect(selectfiles,'bvec',csdprob,'bvec')
	csd_prob.connect(eddycorrect, 'eddy_corrected',csdprob,'in_file')

	csd_prob.connect(csdprob, 'out_file',smooth,'track_file')

	#====================================
	# Running the workflow
	csd_prob.base_dir = os.path.abspath(out_directory)
	csd_prob.write_graph()
	csd_prob.run()

def DTI_calculate_RD(subject_list,base_directory):
	import nibabel as nib
	import os

	for subject in subject_list:
		subject = subject.split('_')[-1]
		dti_folder = base_directory + '_subject_id_' + subject + '/dtifit/'
		if os.path.isdir(dti_folder):
			L1_img  = nib.load(dti_folder + subject + '_L1.nii.gz')
			L1_data = L1_img.get_data()
			L2_img  = nib.load(dti_folder + subject + '_L2.nii.gz')
			L2_data = L2_img.get_data()
			L3_img  = nib.load(dti_folder + subject + '_L3.nii.gz')
			L3_data = L3_img.get_data()
			RD = (L2_data + L2_data)/2
			nib.save(nib.Nifti1Image(RD,L1_img.get_affine()),dti_folder + subject + '_RD.nii.gz')

def CSD_probablistic_tractography_MRTrix(subject_list,base_directory,out_directory):
	#==============================================================
	# Loading required packages
	import nipype.interfaces.io as nio
	import nipype.pipeline.engine as pe
	import nipype.interfaces.utility as util
	import nipype.interfaces.fsl as fsl
	import nipype.interfaces.dipy as dipy
	import nipype.interfaces.mrtrix as mrt
	from own_nipype import DipyDenoise as denoise
	import nipype.interfaces.diffusion_toolkit as dtk
	import nipype.algorithms.misc as misc

	from nipype import SelectFiles
	import os
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
	nodes.append(selectfiles)

	# Denoising
	denoise = pe.Node(interface=denoise(), name='denoise')
	nodes.append(denoise)

	# Eddy-current and motion correction
	eddycorrect = pe.Node(interface=fsl.epi.EddyCorrect(), name='eddycorrect')
	eddycorrect.inputs.ref_num = 0
	nodes.append(eddycorrect)

	# Extract b0 image
	fslroi = pe.Node(interface=fsl.ExtractROI(),name='extract_b0')
	fslroi.inputs.t_min=0
	fslroi.inputs.t_size=1
	nodes.append(fslroi)

	# Fitting the diffusion tensor model
	dwi2tensor = pe.Node(interface=mrt.DWI2Tensor(), name='dwi2tensor')
	tensor2vector = pe.Node(interface=mrt.Tensor2Vector(), name='tensor2vector')
	tensor2adc = pe.Node(interface=mrt.Tensor2ApparentDiffusion(), name='tensor2adc')
	tensor2fa = pe.Node(interface=mrt.Tensor2FractionalAnisotropy(), name='tensor2fa')

	# Create a brain mask
	bet = pe.Node(interface=fsl.BET(frac=0.3,robust=False,mask=True),name='bet')
	nodes.append(bet)

	# Eroding the brain mask
	threshold_b0 = pe.Node(interface=mrt.Threshold(), name='threshold_b0')
	median3d = pe.Node(interface=mrt.MedianFilter3D(), name='median3d')
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
	probCSDstreamtrack = pe.Node(interface=mrt.ProbabilisticSphericallyDeconvolutedStreamlineTrack(), name='probCSDstreamtrack')
	probCSDstreamtrack.inputs.inputmodel = 'SD_PROB'
	probCSDstreamtrack.inputs.desired_number_of_tracks = 150000
	tck2trk = pe.Node(interface=mrt.MRTrix2TrackVis(), name='tck2trk')

	# smoothing the tracts
	smooth = pe.Node(interface=dtk.SplineFilter(step_length=0.5), name='smooth')
	nodes.append(smooth)

	#====================================
	# Setting up the workflow
	csd_prob = pe.Workflow(name='CSD_probablistic_tractography')
	csd_prob.add_nodes(nodes)

	# Reading in files
	csd_prob.connect(infosource, 'subject_id', selectfiles, 'subject_id')

	# Denoising
	csd_prob.connect(selectfiles, 'dwi', denoise, 'in_file')

	# Eddy current and motion correction
	csd_prob.connect(denoise, 'out_file',eddycorrect, 'in_file')
	csd_prob.connect(eddycorrect, 'eddy_corrected', fslroi, 'in_file')

	# Brain extraction
	csd_prob.connect(fslroi, 'roi_file', bet, 'in_file')

	# Creating tensor maps
	csd_prob.connect(selectfiles,'bval',fsl2mrtrix,'bval_file')
	csd_prob.connect(selectfiles,'bvec',fsl2mrtrix,'bvec_file')
	csd_prob.connect(eddycorrect,'eddy_corrected',gunzip,'in_file')
	csd_prob.connect(gunzip,'out_file',dwi2tensor,'in_file')
	csd_prob.connect(fsl2mrtrix,'encoding_file',dwi2tensor,'encoding_file')
	csd_prob.connect(dwi2tensor,'tensor',tensor2vector,'in_file')
	csd_prob.connect(dwi2tensor,'tensor',tensor2adc,'in_file')
	csd_prob.connect(dwi2tensor,'tensor',tensor2fa,'in_file')
	csd_prob.connect(tensor2fa,'FA', MRmult_merge, 'in1')

	# Thresholding to create a mask of single fibre voxels
	csd_prob.connect(gunzip, 'out_file', threshold_b0, 'in_file')
	csd_prob.connect(threshold_b0, 'out_file', median3d, 'in_file')
	csd_prob.connect(median3d, 'out_file', erode_mask_firstpass, 'in_file')
	csd_prob.connect(erode_mask_firstpass, 'out_file', erode_mask_secondpass, 'in_file')
	csd_prob.connect(erode_mask_secondpass,'out_file', MRmult_merge, 'in2')
	csd_prob.connect(MRmult_merge, 'out', MRmultiply,  'in_files')
	csd_prob.connect(MRmultiply, 'out_file', threshold_FA, 'in_file')

	# Create seed mask
	csd_prob.connect(gunzip, 'out_file', gen_WM_mask, 'in_file')
	csd_prob.connect(bet, 'mask_file', gunzip2, 'in_file')
	csd_prob.connect(gunzip2, 'out_file', gen_WM_mask, 'binary_mask')
	csd_prob.connect(fsl2mrtrix, 'encoding_file', gen_WM_mask, 'encoding_file')
	csd_prob.connect(gen_WM_mask, 'WMprobabilitymap', threshold_wmmask, 'in_file')

	# Estimate response
	csd_prob.connect(gunzip, 'out_file', estimateresponse, 'in_file')
	csd_prob.connect(fsl2mrtrix, 'encoding_file', estimateresponse, 'encoding_file')
	csd_prob.connect(threshold_FA, 'out_file', estimateresponse, 'mask_image')

	# CSD calculation
	csd_prob.connect(gunzip, 'out_file', csdeconv, 'in_file')
	csd_prob.connect(gen_WM_mask, 'WMprobabilitymap', csdeconv, 'mask_image')
	csd_prob.connect(estimateresponse, 'response', csdeconv, 'response_file')
	csd_prob.connect(fsl2mrtrix, 'encoding_file', csdeconv, 'encoding_file')

	# Running the tractography
	csd_prob.connect(threshold_wmmask, "out_file", probCSDstreamtrack, "seed_file")
	csd_prob.connect(csdeconv, "spherical_harmonics_image", probCSDstreamtrack, "in_file")
	csd_prob.connect(gunzip, "out_file", tck2trk, "image_file")
	csd_prob.connect(probCSDstreamtrack, "tracked", tck2trk, "in_file")

	# Smoothing the trackfile
	csd_prob.connect(tck2trk, 'out_file',smooth,'track_file')

	#====================================
	# Running the workflow
	csd_prob.base_dir = os.path.abspath(out_directory)
	csd_prob.write_graph()
	csd_prob.run()


def FA_connectome(subject_list,base_directory,out_directory):

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
	from own_nipype import FAconnectome as connectome
	from own_nipype import Extractb0 as extract_b0
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
	probCSDstreamtrack = pe.Node(interface=mrt.ProbabilisticSphericallyDeconvolutedStreamlineTrack(), name='probCSDstreamtrack')
	probCSDstreamtrack.inputs.inputmodel = 'SD_PROB'
	probCSDstreamtrack.inputs.desired_number_of_tracks = 150000
	tck2trk = pe.Node(interface=mrt.MRTrix2TrackVis(), name='tck2trk')

	# smoothing the tracts
	smooth = pe.Node(interface=dtk.SplineFilter(step_length=0.5), name='smooth')

	# Co-registration with MNI space
	mrconvert = pe.Node(mrt.MRConvert(extension='nii'), name='mrconvert')
	flt = pe.Node(interface=fsl.FLIRT(reference=registration_reference, dof=12, cost_func='corratio'), name='flt')

	# Moving tracts to common space
	trkcoreg = pe.Node(interface=trkcoreg(reference=registration_reference),name='trkcoreg')

	# calcuating the connectome matrix
	calc_matrix = pe.Node(interface=connectome(ROI_file='/home/jb07/Desktop/aal.nii.gz'),name='calc_matrix')

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
	fa_connectome.connect(threshold_wmmask, "out_file", probCSDstreamtrack, "seed_file")
	fa_connectome.connect(csdeconv, "spherical_harmonics_image", probCSDstreamtrack, "in_file")
	fa_connectome.connect(gunzip, "out_file", tck2trk, "image_file")
	fa_connectome.connect(probCSDstreamtrack, "tracked", tck2trk, "in_file")

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

	#====================================
	# Running the workflow
	fa_connectome.base_dir = os.path.abspath(out_directory)
	fa_connectome.write_graph()
	fa_connectome.run(plugin='MultiProc')
