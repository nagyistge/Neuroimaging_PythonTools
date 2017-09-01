# Inputs
subject_list = ['CBU150094']
base_directory = '/imaging/jb07/CALM/CALM_BIDS/'
out_directory = '/home/jb07/Desktop/test/'

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

# Defining the output
datasink = pe.Node(nio.DataSink(), name='sinker')
datasink.inputs.base_directory = '/home/jb07/test'
nodes.append(datasink)

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


