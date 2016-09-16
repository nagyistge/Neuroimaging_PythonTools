def FreeSurfer_Pipeline(subject_list, base_directory, out_directory, template_directory):

    # Loading required packages
    from nipype.interfaces.ants import N4BiasFieldCorrection
    from nipype.interfaces.ants.segmentation import BrainExtraction
    import nipype.interfaces.dipy as dipy
    from nipype.interfaces.freesurfer import ReconAll
    import nipype.interfaces.fsl as fsl
    import nipype.interfaces.io as nio
    import nipype.interfaces.utility as util
    from nipype import SelectFiles
    import nipype.pipeline.engine as pe
    from own_nipype import AdditionalDTIMeasures
    from own_nipype import DipyDenoise
    from own_nipype import DipyDenoiseT1
    from own_nipype import FSRename
    from own_nipype import FS_Gyrification
    import os

    # ====================================
    # Defining the nodes for the workflow

    # Getting the subject ID
    infosource  = pe.Node(interface=util.IdentityInterface(fields = ['subject_id']), name='infosource')
    infosource.iterables = ('subject_id', subject_list)

    # Getting the relevant data
    templates = dict(T1='{subject_id}/anat/{subject_id}_T1w.nii.gz', dwi='{subject_id}/dwi/{subject_id}_dwi.nii.gz', bvec='{subject_id}/dwi/{subject_id}_dwi.bvec', bval='{subject_id}/dwi/{subject_id}_dwi.bval')

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

    gyrification = pe.Node(interface=FS_Gyrification(), name='gyrification')

    # DataSink
    datasink = pe.Node(interface=nio.DataSink(), name='datasink')
    datasink.inputs.parameterization = False
    datasink.inputs.base_directory = out_directory

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

    # ==================================================================
    # Transformations to common space

    # Moving T1 to template space
    T1_to_template_flirt = pe.Node(interface=fsl.FLIRT(), name='T1_to_template_flirt')
    T1_to_template_flirt.inputs.cost_func = 'corratio'
    T1_to_template_flirt.inputs.dof = 6
    T1_to_template_flirt.inputs.out_matrix_file = 'subjectT1_to_template.mat'
    T1_to_template_flirt.inputs.reference = template_directory + '/T_template_BrainCerebellum.nii.gz'

    # Moving template space to MNI space
    template_to_MNI_flirt = pe.Node(interface=fsl.FLIRT(), name='template_to_MNI_flirt')
    template_to_MNI_flirt.inputs.cost_func = 'corratio'
    template_to_MNI_flirt.inputs.dof = 6
    template_to_MNI_flirt.inputs.out_matrix_file = 'templateT1_to_MNI.mat'
    template_to_MNI_flirt.inputs.reference = os.environ['FSLDIR'] + '/data/standard/MNI152_T1_1mm.nii.gz'

    # Moving dwi to template space
    dwi_to_template_flirt = pe.Node(interface=fsl.FLIRT(), name='dwi_to_template_flirt')
    dwi_to_template_flirt.inputs.cost_func = 'mutualinfo'
    dwi_to_template_flirt.inputs.dof = 6
    dwi_to_template_flirt.inputs.out_matrix_file = 'subjectDWI_to_template.mat'
    dwi_to_template_flirt.inputs.reference = template_directory + '/T_template_BrainCerebellum.nii.gz'

    dwi_template_to_MNI_flirt = pe.Node(interface=fsl.FLIRT(), name='dwi_template_to_MNI_flirt')
    dwi_template_to_MNI_flirt.inputs.cost_func = 'mutualinfo'
    dwi_template_to_MNI_flirt.inputs.dof = 6
    dwi_template_to_MNI_flirt.inputs.out_matrix_file = 'templateDWI_to_MNI.mat'
    dwi_template_to_MNI_flirt.inputs.reference = os.environ['FSLDIR'] + '/data/standard/MNI152_T1_1mm.nii.gz'

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
    freesurfer_pipeline.connect(autorecon1, 'subject_id', autorecon2, 'subject_id')
    freesurfer_pipeline.connect(autorecon1, 'subjects_dir', autorecon2, 'subjects_dir')
    freesurfer_pipeline.connect(autorecon1, 'subject_id', rename, 'subject_id')
    freesurfer_pipeline.connect(autorecon2, 'subject_id', autorecon3, 'subject_id')
    freesurfer_pipeline.connect(autorecon2, 'subjects_dir', autorecon3, 'subjects_dir')
    freesurfer_pipeline.connect(autorecon3, 'subject_id', gyrification, 'subject_id')
    freesurfer_pipeline.connect(autorecon3, 'subjects_dir', gyrification, 'subjects_dir')
    freesurfer_pipeline.connect(brainextraction, 'BrainExtractionBrain', T1_to_template_flirt, 'in_file')
    freesurfer_pipeline.connect(T1_to_template_flirt, 'out_file', template_to_MNI_flirt, 'in_file')

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
    freesurfer_pipeline.connect(dtifit, 'FA', dwi_to_template_flirt, 'in_file')
    freesurfer_pipeline.connect(dwi_to_template_flirt, 'out_file', dwi_template_to_MNI_flirt, 'in_file')

    # Moving files to the output directory
    freesurfer_pipeline.connect(infosource, 'subject_id', datasink, 'container')
    freesurfer_pipeline.connect(dtifit, 'FA', datasink, 'diffusion.@FA')
    freesurfer_pipeline.connect(dtifit, 'MD', datasink, 'diffusion.@MD')
    freesurfer_pipeline.connect(get_rd, 'RD', datasink, 'diffusion.@RD')
    freesurfer_pipeline.connect(get_rd, 'AD', datasink, 'diffusion.@AD')
    freesurfer_pipeline.connect(dwi_to_template_flirt, 'out_matrix_file', datasink, 'diffusion.@dwi_to_template')
    freesurfer_pipeline.connect(dwi_template_to_MNI_flirt, 'out_matrix_file', datasink, 'diffusion.@template_to_MNI')

    freesurfer_pipeline.connect(autorecon3, 'curv', datasink, 'surface.@curvature')
    freesurfer_pipeline.connect(autorecon3, 'thickness', datasink, 'surface.@thickness')
    freesurfer_pipeline.connect(autorecon3, 'sulc', datasink, 'surface.@sulcal_depth')
    freesurfer_pipeline.connect(autorecon3, 'volume', datasink, 'surface.@volume')
    freesurfer_pipeline.connect(gyrification, 'lh_gyr', datasink, 'surface.@lh_gyrification')
    freesurfer_pipeline.connect(gyrification, 'rh_gyr', datasink, 'surface.@rh_gyrification')
    freesurfer_pipeline.connect(T1_to_template_flirt, 'out_matrix_file', datasink, 'surface.@T1_to_template')
    freesurfer_pipeline.connect(template_to_MNI_flirt, 'out_matrix_file', datasink, 'surface.@template_to_MNI')

    # ==================================================================
    # Running the workflow
    freesurfer_pipeline.base_dir = os.path.abspath(out_directory)
    freesurfer_pipeline.write_graph()
    freesurfer_pipeline.run(plugin='PBSGraph')

import pandas as pd
dwi_df = pd.read_csv('/imaging/jb07/CALM/DWI/motion_estimation/Displacements_Results.csv')
dwi_df.columns = ['MRI.ID', 'movement']
dwi_df = dwi_df[dwi_df['movement'] < 3]
T1_df = pd.read_csv('/imaging/jb07/CALM/T1/T1_useable_data.csv')
T1_df = T1_df[T1_df['useable'] == 1]
df = pd.merge(dwi_df, T1_df, on='MRI.ID')
subject_list = sorted(df['MRI.ID'].values)

out_directory = '/imaging/jb07/CALM/BCNI_Morphometry/'
base_directory = '/imaging/jb07/CALM/CALM_BIDS/'
template_directory = '/imaging/jb07/Atlases/NKI/'
subject_list = subject_list[1:10]
FreeSurfer_Pipeline(subject_list, base_directory, out_directory, template_directory)
