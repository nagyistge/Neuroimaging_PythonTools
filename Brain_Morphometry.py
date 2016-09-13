def dwi_preproc(subject_list, base_directory, out_directory, template_directory):
    # Loading required packages
    import nipype.pipeline.engine as pe
    import nipype.interfaces.utility as util
    import nipype.interfaces.fsl as fsl
    import nipype.interfaces.dipy as dipy
    from own_nipype import DipyDenoise as denoise
    from own_nipype import AdditionalDTIMeasures
    from own_nipype import ants_QuickSyN
    from nipype.interfaces.ants.segmentation import CorticalThickness
    from nipype import SelectFiles
    import os


    #====================================
    # Defining the nodes for the workflow

    # Getting the subject ID
    infosource  = pe.Node(interface=util.IdentityInterface(fields=['subject_id']), name='infosource')
    infosource.iterables = ('subject_id', subject_list)

    # Getting the relevant diffusion-weighted data
    templates = dict(T1='{subject_id}/anat/{subject_id}_T1w.nii.gz',
                    dwi='{subject_id}/dwi/{subject_id}_dwi.nii.gz',
                    bvec='{subject_id}/dwi/{subject_id}_dwi.bvec',
                    bval='{subject_id}/dwi/{subject_id}_dwi.bval')

    selectfiles = pe.Node(SelectFiles(templates),
                       name="selectfiles")
    selectfiles.inputs.base_directory = os.path.abspath(base_directory)

    #====================================
    # Diffusion-data
    # Denoising
    denoise = pe.Node(interface=denoise(), name='denoise')

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

    #====================================
    # T1-weighted data
    corticalthickness = pe.Node(interface=CorticalThickness(), name='corticalthickness')
    corticalthickness.inputs.dimension = 3
    corticalthickness.inputs.brain_template = template_directory + '/T_template.nii.gz'
    corticalthickness.inputs.brain_probability_mask = template_directory + '/T_template_BrainCerebellumProbabilityMask.nii.gz'
    corticalthickness.inputs.segmentation_priors = sorted([template_directory + '/Priors/' + prior for prior in os.listdir(template_directory + '/Priors/')])
    corticalthickness.inputs.t1_registration_template = template_directory + '/T_template_BrainCerebellum.nii.gz'


    #====================================
    # Setting up the workflow
    dwi_preproc = pe.Workflow(name='dwi_preproc')

    dwi_preproc.connect(infosource, 'subject_id', selectfiles, 'subject_id')

    # Diffusion data
    dwi_preproc.connect(selectfiles, 'dwi', denoise, 'in_file')
    dwi_preproc.connect(denoise, 'out_file', eddycorrect, 'in_file')
    dwi_preproc.connect(eddycorrect, 'eddy_corrected', resample, 'in_file')
    dwi_preproc.connect(resample, 'out_file', fslroi, 'in_file')
    dwi_preproc.connect(fslroi, 'roi_file', bet, 'in_file')
    dwi_preproc.connect(infosource, 'subject_id', dtifit, 'base_name')
    dwi_preproc.connect(resample, 'out_file', dtifit, 'dwi')
    dwi_preproc.connect(selectfiles, 'bvec', dtifit, 'bvecs')
    dwi_preproc.connect(selectfiles, 'bval', dtifit, 'bvals')
    dwi_preproc.connect(bet, 'mask_file', dtifit, 'mask')
    dwi_preproc.connect(dtifit, 'L1', get_rd, 'L1')
    dwi_preproc.connect(dtifit, 'L2', get_rd, 'L2')
    dwi_preproc.connect(dtifit, 'L3', get_rd, 'L3')
    dwi_preproc.connect(dtifit, 'FA', quicksyn, 'moving_image')

    # T1w data
    dwi_preproc.connect(selectfiles, 'T1', corticalthickness, 'anatomical_image')

    #====================================
    # Running the workflow
    dwi_preproc.base_dir = os.path.abspath(out_directory)
    dwi_preproc.write_graph()
    dwi_preproc.run(plugin='PBSGraph')

out_directory = '/imaging/jb07/CALM/BCNI_Morphometry/'
base_directory = '/imaging/jb07/CALM/CALM_BIDS/'
template_directory = '/imaging/jb07/Atlases/NKI/'
subject_list = ['CBU150084']
dwi_preproc(subject_list, base_directory, out_directory, template_directory)
