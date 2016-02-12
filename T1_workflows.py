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
	templates = dict(T1='{subject_id}/{subject_id}_T1w.nii.gz')

	selectfiles = pe.Node(SelectFiles(templates),
	                   name="selectfiles")
	selectfiles.inputs.base_directory = os.path.abspath(base_directory)
	nodes.append(selectfiles)


	reconall = pe.Node(interface=ReconAll(), name='reconall')
	reconall.inputs.directive = 'all'
	reconall.inputs.subjects_dir = out_directory

	# Setting up the workflow
	fs_reconall = pe.Workflow(name='fs_reconall')

	# Reading in files
	fs_reconall.connect(infosource, 'subject_id', selectfiles, 'subject_id')
	fs_reconall.connect(selectfiles, 'T1', reconall, 'T1_files')
	fs_reconall.connect(infosource, 'subject_id', reconall, 'subject_id')


	# Running the workflow
	fs_reconall.base_dir = os.path.abspath(out_directory)
	fs_reconall.write_graph()
	fs_reconall.run(plugin='MultiProc')

