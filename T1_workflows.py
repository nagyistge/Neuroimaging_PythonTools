def FreeSurfer_Reconall(subject_list,base_directory,out_directory):
	#==============================================================
	# Loading required packages
	import nipype.interfaces.io as nio
	import nipype.pipeline.engine as pe
	import nipype.interfaces.utility as util
	from nipype.interfaces.freesurfer import ReconAll
	from nipype import SelectFiles
	import os
	nodes = list()

	#====================================
	# Defining the nodes for the workflow

	# Getting the subject ID
	infosource  = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),name='infosource')
	infosource.iterables = ('subject_id', subject_list)

	# Getting the relevant diffusion-weighted data
	templates = dict(T1='{subject_id}/anat/{subject_id}_T1w.nii.gz')

	selectfiles = pe.Node(SelectFiles(templates),
	                   name="selectfiles")
	selectfiles.inputs.base_directory = os.path.abspath(base_directory)
	nodes.append(selectfiles)


	reconall = pe.Node(interface=ReconAll(), name='reconall')
	reconall.inputs.directive = 'autorecon2'
	reconall.inputs.subjects_dir = out_directory
	reconall.inputs.flags = '-no-isrunning'
	reconall.inputs.ignore_exception = True

	# Setting up the workflow
	fs_reconall = pe.Workflow(name='fs_reconall')

	# Reading in files
	fs_reconall.connect(infosource, 'subject_id', selectfiles, 'subject_id')
	fs_reconall.connect(selectfiles, 'T1', reconall, 'T1_files')
	fs_reconall.connect(infosource, 'subject_id', reconall, 'subject_id')


	# Running the workflow
	fs_reconall.base_dir = os.path.abspath(out_directory)
	fs_reconall.write_graph()
	fs_reconall.run('PBSGraph')

def run_FSL_anat(subject_list,base_directory,out_directory):
	#==============================================================
	# Loading required packages
	import nipype.interfaces.io as nio
	import nipype.pipeline.engine as pe
	import nipype.interfaces.utility as util
	from own_nipype import FSLANAT
	from nipype import SelectFiles
	import os
	nodes = list()

	#====================================
	# Defining the nodes for the workflow

	# Getting the subject ID
	infosource  = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),name='infosource')
	infosource.iterables = ('subject_id', subject_list)

	# Getting the relevant diffusion-weighted data
	templates = dict(T1='{subject_id}/anat/{subject_id}_T1w.nii.gz')

	selectfiles = pe.Node(SelectFiles(templates),
	                   name="selectfiles")
	selectfiles.inputs.base_directory = os.path.abspath(base_directory)
	nodes.append(selectfiles)

	fslanat = pe.Node(interface=FSLANAT(out_directory=out_directory), name='fslanat')

	# Setting up the workflow
	fsl_anat = pe.Workflow(name='FSL_anat')

	# Reading in files
	fsl_anat.connect(infosource, 'subject_id', selectfiles, 'subject_id')
	fsl_anat.connect(selectfiles, 'T1', fslanat, 'in_file')

	# Running the workflow
	fsl_anat.base_dir = os.path.abspath(out_directory)
	fsl_anat.write_graph()
	fsl_anat.run('MultiProc')

def GM_covariance_matrix(subject_list,base_directory,out_directory):
	#==============================================================
	# Loading required packages
	import nipype.interfaces.io as nio
	import nipype.pipeline.engine as pe
	import nipype.interfaces.utility as util
	from nipype import SelectFiles
	import nipype.interfaces.fsl as fsl
	import os
	nodes = list()

	reference = os.environ['FSLDIR'] + '/data/standard/MNI152_T1_1mm_brain.nii.gz'

	#====================================
	# Defining the nodes for the workflow

	# Getting the subject ID
	infosource  = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),name='infosource')
	infosource.iterables = ('subject_id', subject_list)

	# Getting the relevant diffusion-weighted data
	templates = dict(T1='{subject_id}/anat/{subject_id}_T1w.nii.gz')

	selectfiles = pe.Node(SelectFiles(templates),
	                   name="selectfiles")
	selectfiles.inputs.base_directory = os.path.abspath(base_directory)
	nodes.append(selectfiles)

	# Removing non-brain tissue
	bet = pe.Node(interface=fsl.BET(),name='bet')

	# Transforming the T1 images to MNI space using FLIRT
	flt = pe.Node(interface=fsl.FLIRT(dof=12,cost_func='corratio',reference=reference),name='flt')

	# Setting up the workflow
	covar_mat = pe.Workflow(name='Covar_Matrix')

	# Reading in files
	covar_mat.connect(infosource, 'subject_id', selectfiles, 'subject_id')
	covar_mat.connect(selectfiles, 'T1', bet, 'in_file')
	covar_mat.connect(bet, 'out_file', flt, 'in_file')

	# Running the workflow
	covar_mat.base_dir = os.path.abspath(out_directory)
	covar_mat.write_graph()
	covar_mat.run('PBSGraph')