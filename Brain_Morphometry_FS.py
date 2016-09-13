def FreeSurfer_Pipeline(subject_list, base_directory, out_directory, template_directory):

    # Loading required packages
    import nipype.pipeline.engine as pe
    import nipype.interfaces.utility as util
    from nipype.interfaces.ants import N4BiasFieldCorrection
    from nipype.interfaces.ants.segmentation import BrainExtraction
    from nipype.interfaces.freesurfer import ReconAll
    from nipype import SelectFiles
    from own_nipype import FSRename
    import os

    # ====================================
    # Defining the nodes for the workflow

    # Getting the subject ID
    infosource  = pe.Node(interface=util.IdentityInterface(fields = ['subject_id']), name='infosource')
    infosource.iterables = ('subject_id', subject_list)

    # Getting the relevant diffusion-weighted data
    templates = dict(T1='{subject_id}/anat/{subject_id}_T1w.nii.gz')

    selectfiles = pe.Node(SelectFiles(templates),
                       name="selectfiles")
    selectfiles.inputs.base_directory = os.path.abspath(base_directory)

    # Bias field correction
    n4 = pe.Node(interface=N4BiasFieldCorrection(), name='n4')
    n4.inputs.dimension = 3
    n4.inputs.save_bias = True

    # Brain extraction
    brainextraction = pe.Node(interface=BrainExtraction(), name='brainextraction')
    brainextraction.inputs.dimension = 3
    brainextraction.inputs.brain_template = template_directory + '/T_template.nii.gz'
    brainextraction.inputs.brain_probability_mask = template_directory + '/T_template_BrainCerebellumProbabilityMask.nii.gz'

    # Renaming files for FreeSurfer
    rename = pe.Node(FSRename(), name='rename')
    rename.inputs.out_directory = out_directory

    # Running FreeSurfer
    autorecon1 = pe.Node(interface=ReconAll(), name='autorecon1')
    autorecon1.inputs.directive = 'autorecon1'
    autorecon1.inputs.args = '-noskullstrip'

    autorecon2 = pe.Node(interface=ReconAll(), name='autorecon2')
    autorecon2.inputs.directive = 'autorecon2'

    autorecon3 = pe.Node(interface=ReconAll(), name='autorecon3')
    autorecon3.inputs.directive = 'autorecon3'

    # ====================================
    # Setting up the workflow
    freesurfer_pipeline = pe.Workflow(name='freesurfer_pipeline')
    freesurfer_pipeline.connect(infosource, 'subject_id', selectfiles, 'subject_id')
    freesurfer_pipeline.connect(selectfiles, 'T1', n4, 'input_image')
    freesurfer_pipeline.connect(n4, 'output_image', brainextraction, 'anatomical_image')

    freesurfer_pipeline.connect(infosource, 'subject_id', autorecon1, 'subject_id')
    freesurfer_pipeline.connect(brainextraction, 'BrainExtractionBrain', autorecon1, 'T1_files')
    freesurfer_pipeline.connect(infosource, 'subject_id', rename, 'subject_id')
    freesurfer_pipeline.connect(autorecon1, 'subject_id', autorecon2, 'subject_id')
    freesurfer_pipeline.connect(autorecon1, 'subjects_dir', autorecon2, 'subjects_dir')
    freesurfer_pipeline.connect(autorecon2, 'subject_id', autorecon3, 'subject_id')
    freesurfer_pipeline.connect(autorecon2, 'subjects_dir', autorecon3, 'subjects_dir')
    #====================================
    # Running the workflow
    freesurfer_pipeline.base_dir = os.path.abspath(out_directory)
    freesurfer_pipeline.write_graph()
    freesurfer_pipeline.run()


out_directory = '/imaging/jb07/CALM/BCNI_Morphometry/'
base_directory = '/imaging/jb07/CALM/CALM_BIDS/'
template_directory = '/imaging/jb07/Atlases/NKI/'
subject_list = ['CBU150084']
FreeSurfer_Pipeline(subject_list, base_directory, out_directory, template_directory)





