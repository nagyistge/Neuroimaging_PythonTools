def FreeSurfer_Pipeline(subject_list, base_directory, out_directory, template_directory):

    # Loading required packages
    from nipype.interfaces.ants import N4BiasFieldCorrection
    from nipype.interfaces.ants.segmentation import BrainExtraction
    from nipype.interfaces.freesurfer import ReconAll
    import nipype.interfaces.fsl as fsl
    import nipype.interfaces.dipy as dipy
    import nipype.interfaces.utility as util
    from nipype import SelectFiles
    import nipype.pipeline.engine as pe
    from own_nipype import AdditionalDTIMeasures
    from own_nipype import DipyDenoise
    from own_nipype import DipyDenoiseT1
    from own_nipype import FSRename
    from own_nipype import ants_QuickSyN
    import os

    # ====================================
    # Defining the nodes for the workflow

    # Getting the subject ID
    infosource  = pe.Node(interface=util.IdentityInterface(fields = ['subject_id']), name='infosource')
    infosource.iterables = ('subject_id', subject_list)

    # Getting the relevant data
    templates = dict(T1='{subject_id}/anat/{subject_id}_T1w.nii.gz')

    selectfiles = pe.Node(SelectFiles(templates), name="selectfiles")
    selectfiles.inputs.base_directory = os.path.abspath(base_directory)

    #===================================================================
    # Processing of T1-weighted data

    # Getting a better field of view
    robustfov = pe.Node(interface=fsl.RobustFOV(), name='robustfov')

    # Denoising
    T1_denoise = pe.Node(interface=DipyDenoiseT1(), name='T1_denoise')

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

    # ==================================================================
    # Processing of diffusion-weighted data
    # Denoising
    dwi_denoise = pe.Node(interface=DipyDenoise(), name='dwi_denoise')

    # Eddy-current and motion correction
    eddycorrect = pe.Node(interface=fsl.epi.EddyCorrect(), name='eddycorrect')
    eddycorrect.inputs.ref_num = 0

    # Upsampling
    resample = pe.Node(interface=dipy.Resample(interp=3,vox_size=(1., 1., 1.)), name='resample')

    # Extract b0 image
    fslroi = pe.Node(interface=fsl.ExtractROI(), name='extract_b0')
    fslroi.inputs.t_min=0
    fslroi.inputs.t_size=1

    # Create a brain mask
    bet = pe.Node(interface=fsl.BET(frac=0.3,robust=False,mask=True), name='bet')

    # Fitting the diffusion tensor model
    dtifit = pe.Node(interface=fsl.DTIFit(), name='dtifit')

    # Getting AD and RD
    get_rd = pe.Node(interface=AdditionalDTIMeasures(), name='get_rd')

    # Calculating transform from subject to common space
    quicksyn = pe.Node(interface=ants_QuickSyN(), name='quicksyn')
    quicksyn.inputs.fixed_image = template_directory + '/T_template_BrainCerebellum.nii.gz'
    quicksyn.inputs.image_dimensions = 3
    quicksyn.inputs.output_prefix = 'NKIspace_'
    quicksyn.inputs.transform_type = 's'

    # ==================================================================
    # Processing of diffusion-weighted data

    # ==================================================================
    # Setting up the workflow
    freesurfer_pipeline = pe.Workflow(name='freesurfer_pipeline')
    freesurfer_pipeline.connect(infosource, 'subject_id', selectfiles, 'subject_id')

    # T1 data
    freesurfer_pipeline.connect(selectfiles, 'T1', robustfov, 'in_file')
    freesurfer_pipeline.connect(robustfov, 'out_roi', T1_denoise, 'in_file')
    freesurfer_pipeline.connect(T1_denoise, 'out_file', n4, 'input_image')
    freesurfer_pipeline.connect(n4, 'output_image', brainextraction, 'anatomical_image')
    freesurfer_pipeline.connect(infosource, 'subject_id', autorecon1, 'subject_id')
    freesurfer_pipeline.connect(brainextraction, 'BrainExtractionBrain', autorecon1, 'T1_files')
    freesurfer_pipeline.connect(infosource, 'subject_id', rename, 'subject_id')
    freesurfer_pipeline.connect(autorecon1, 'subject_id', autorecon2, 'subject_id')
    freesurfer_pipeline.connect(autorecon1, 'subjects_dir', autorecon2, 'subjects_dir')
    freesurfer_pipeline.connect(autorecon2, 'subject_id', autorecon3, 'subject_id')
    freesurfer_pipeline.connect(autorecon2, 'subjects_dir', autorecon3, 'subjects_dir')

    # Diffusion data
    freesurfer_pipeline.connect(selectfiles, 'dwi', dwi_denoise, 'in_file')
    freesurfer_pipeline.connect(dwi_denoise, 'out_file', eddycorrect, 'in_file')
    freesurfer_pipeline.connect(eddycorrect, 'eddy_corrected', resample, 'in_file')
    freesurfer_pipeline.connect(resample, 'out_file', fslroi, 'in_file')
    freesurfer_pipeline.connect(fslroi, 'roi_file', bet, 'in_file')
    freesurfer_pipeline.connect(infosource, 'subject_id', dtifit, 'base_name')
    freesurfer_pipeline.connect(resample, 'out_file', dtifit, 'dwi')
    freesurfer_pipeline.connect(selectfiles, 'bvec', dtifit, 'bvecs')
    freesurfer_pipeline.connect(selectfiles, 'bval', dtifit, 'bvals')
    freesurfer_pipeline.connect(bet, 'mask_file', dtifit, 'mask')
    freesurfer_pipeline.connect(dtifit, 'L1', get_rd, 'L1')
    freesurfer_pipeline.connect(dtifit, 'L2', get_rd, 'L2')
    freesurfer_pipeline.connect(dtifit, 'L3', get_rd, 'L3')
    freesurfer_pipeline.connect(dtifit, 'FA', quicksyn, 'moving_image')

    # ==================================================================
    # Running the workflow
    freesurfer_pipeline.base_dir = os.path.abspath(out_directory)
    freesurfer_pipeline.write_graph()
    freesurfer_pipeline.run()


out_directory = '/imaging/jb07/CALM/BCNI_Morphometry/'
base_directory = '/imaging/jb07/CALM/CALM_BIDS/'
template_directory = '/imaging/jb07/Atlases/NKI/'
subject_list = ['CBU150084']
FreeSurfer_Pipeline(subject_list, base_directory, out_directory, template_directory)
