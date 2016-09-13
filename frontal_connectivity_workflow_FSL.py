def frontal_connectivity_FSL(subject_list, base_directory, out_directory):

    # Loading required packages
    import nibabel as nib
    from nipype.interfaces.ants import N4BiasFieldCorrection
    import nipype.interfaces.fsl as fsl
    import nipype.interfaces.utility as util
    from nipype import SelectFiles
    import nipype.pipeline.engine as pe
    import nipype.algorithms.misc as misc
    import os
    from own_nipype import ImageOverlap, DipyDenoise

    # Workflow
    # Getting the subject ID
    infosource = pe.Node(interface=util.IdentityInterface(
        fields=['subject_id']), name='infosource')
    infosource.iterables = ('subject_id', subject_list)

    # Getting the relevant data
    templates = dict(T1='{subject_id}/anat/{subject_id}_T1w.nii.gz',
                     dwi='{subject_id}/dwi/{subject_id}_dwi.nii.gz',
                     bvec='{subject_id}/dwi/{subject_id}_dwi.bvec',
                     bval='{subject_id}/dwi/{subject_id}_dwi.bval')

    selectfiles = pe.Node(SelectFiles(templates),
                          name="selectfiles")
    selectfiles.inputs.base_directory = os.path.abspath(base_directory)

    # Processing of T1w images
    # Centering the image
    robustfov = pe.Node(interface=fsl.RobustFOV(), name='robustfov')

    # Bias-field correction
    n4 = pe.Node(interface=N4BiasFieldCorrection(dimension=3), name='n4')
    n4.inputs.save_bias = True

    # Calculate brain mask
    T1_bet = pe.Node(interface=fsl.BET(robust=False, mask=True), name='T1_bet')

    # Image segmentation
    fast = pe.Node(interface=fsl.FAST(), name='fast')
    fast.inputs.img_type = 1
    fast.inputs.no_bias = True

    # Select files from the partial volume segmentation
    select1 = pe.Node(interface=util.Select(index=1), name='select1')
    select2 = pe.Node(interface=util.Select(index=2), name='select2')

    # Getting a mask of the white matter/grey matter interface
    overlap = pe.Node(interface=ImageOverlap(), name='overlap')

    # Creating a mask of frontal regions
    atlas_file = os.environ['FSLDIR'] + '/data/atlases/HarvardOxford/' + \
        'HarvardOxford-cort-maxprob-thr0-1mm.nii.gz'

    if not os.path.isdir(out_directory + '/frontal_connectivity_FSL/'):
        os.mkdir(out_directory + '/frontal_connectivity_FSL/')

    img = nib.load(atlas_file)
    atlas_data = img.get_data()
    atlas_affine = img.get_affine()
    atlas_data[atlas_data > 6] = 0
    atlas_data[atlas_data == 2] = 0
    atlas_data[atlas_data > 0] = 1
    nib.save(nib.Nifti1Image(atlas_data, atlas_affine),
             out_directory + '/frontal_connectivity_FSL/frontalROIs.nii.gz')

    img = nib.load(atlas_file)
    atlas_data = img.get_data()
    atlas_data[atlas_data == 2] = 99
    atlas_data[atlas_data < 6] = 0
    atlas_data[atlas_data > 0] = 1
    number_of_voxels = len(atlas_data[atlas_data == 1])
    nib.save(nib.Nifti1Image(atlas_data, atlas_affine),
             out_directory + '/frontal_connectivity_FSL/targetROIs.nii.gz')

    frontalROI = pe.Node(interface=util.Function(
        input_names=['out_directory'],
        output_names=['frontalROI']),
        name='frontalROI')

    frontalROI.inputs.function_str = "def pass_filename(out_directory):   return out_directory + '/frontal_connectivity_FSL/frontalROIs.nii.gz'"
    frontalROI.inputs.out_directory = out_directory

    targetROI = pe.Node(interface=util.Function(
        input_names=['out_directory'],
        output_names=['targetROI']),
        name='targetROI')

    targetROI.inputs.function_str = "def pass_filename(out_directory):    return out_directory + '/frontal_connectivity_FSL/targetROIs.nii.gz'"
    targetROI.inputs.out_directory = out_directory

    # Processing of diffusion data
    denoise = pe.Node(interface=DipyDenoise(), name='denoise')

    # Eddy-current and motion correction
    eddycorrect = pe.Node(interface=fsl.epi.EddyCorrect(), name='eddycorrect')
    eddycorrect.inputs.ref_num = 0

    # Extract b0 image
    fslroi = pe.Node(interface=fsl.ExtractROI(), name='extract_b0')
    fslroi.inputs.t_min = 0
    fslroi.inputs.t_size = 1

    # Create a brain mask
    dwi_bet = pe.Node(interface=fsl.BET(
        frac=0.3, robust=False, mask=True), name='dwi_bet')

    # Fitting the diffusion tensor model
    dti = pe.Node(interface=fsl.DTIFit(), name='dti')

    # Fitting the BEDPOST model
    bedp = pe.Node(interface=fsl.BEDPOSTX5(n_fibres=1), name='bedp')

    # Probabilistic tractography
    pbx = pe.Node(interface=fsl.ProbTrackX(), name='pbx')
    pbx.inputs.n_samples=5 #5000
    pbx.inputs.seed =  out_directory + '/frontal_connectivity_FSL/frontalROIs.nii.gz'
    pbx.inputs.target_masks = out_directory + '/frontal_connectivity_FSL/targetROIs.nii.gz'
    pbx.inputs.c_thresh = 0.2
    pbx.inputs.n_steps=2000
    pbx.inputs.step_length=0.5
    pbx.inputs.opd=True
    pbx.inputs.os2t=True
    pbx.inputs.loop_check=True

    # Moving ribbon to dwi space
    T1_to_dwi_flt = pe.Node(interface=fsl.FLIRT(
        cost_func='mutualinfo'), name='T1_to_dwi_flt')
    T1_to_dwi_applyxfm = pe.Node(fsl.ApplyXfm(
        apply_xfm=True), name='T1_to_dwi_applyxfm')

    # Moving the mask to subject space
    MNI_to_dwi_flt = pe.Node(interface=fsl.FLIRT(
        cost_func='mutualinfo'), name='MNI_to_dwi_flt')
    MNI_to_dwi_flt.inputs.in_file = os.environ[
        'FSLDIR'] + '/data/standard/MNI152_T1_1mm_brain.nii.gz'

    MNI_to_dwi_applyxfm_seed = pe.Node(fsl.ApplyXfm(
        apply_xfm=True), name='MNI_to_dwi_applyxfm_seed')
    MNI_to_dwi_threshold_seed = pe.Node(interface=fsl.maths.Threshold(
        thresh=0.9), name='MNI_to_dwi_threshold_seed')

    MNI_to_dwi_applyxfm_target = pe.Node(fsl.ApplyXfm(
        apply_xfm=True), name='MNI_to_dwi_applyxfm_target')
    MNI_to_dwi_threshold_target = pe.Node(interface=fsl.maths.Threshold(
        thresh=0.9), name='MNI_to_dwi_threshold_target')

    # Multiplying the masks
    multiply = pe.Node(interface=fsl.maths.BinaryMaths(
        operation='mul'), name='multiply')
    multiply2 = pe.Node(interface=fsl.maths.BinaryMaths(
        operation='mul'), name='multiply2')

    # Moving the density image to common space
    invt = pe.Node(interface=fsl.ConvertXFM(invert_xfm=True), name='invt')
    dwi_to_MNI_applyxfm = pe.Node(fsl.ApplyXfm(
        apply_xfm=True), name='dwi_to_MNI_applyxfm')
    dwi_to_MNI_applyxfm.inputs.reference = os.environ[
        'FSLDIR'] + '/data/standard/MNI152_T1_1mm_brain.nii.gz'

    # Setting up the workflow
    frontal_connectivity_FSL = pe.Workflow(name='frontal_connectivity_FSL')

    # T1 processing
    frontal_connectivity_FSL.connect(
        infosource, 'subject_id', selectfiles, 'subject_id')
    frontal_connectivity_FSL.connect(selectfiles, 'T1', robustfov, 'in_file')
    frontal_connectivity_FSL.connect(robustfov, 'out_roi', T1_bet, 'in_file')
    frontal_connectivity_FSL.connect(T1_bet, 'out_file', n4, 'input_image')
    frontal_connectivity_FSL.connect(n4, 'output_image', fast, 'in_files')
    frontal_connectivity_FSL.connect(
        fast, 'partial_volume_files', select1, 'inlist')
    frontal_connectivity_FSL.connect(
        fast, 'partial_volume_files', select2, 'inlist')
    frontal_connectivity_FSL.connect(select1, 'out', overlap, 'in_file1')
    frontal_connectivity_FSL.connect(select2, 'out', overlap, 'in_file2')

    # Getting the ROI into dwi space
    frontal_connectivity_FSL.connect(n4, 'output_image', T1_to_dwi_flt, 'in_file')
    frontal_connectivity_FSL.connect(
        fslroi, 'roi_file', T1_to_dwi_flt, 'reference')
    frontal_connectivity_FSL.connect(
        overlap, 'out_file', T1_to_dwi_applyxfm, 'in_file')
    frontal_connectivity_FSL.connect(
        fslroi, 'roi_file', T1_to_dwi_applyxfm, 'reference')
    frontal_connectivity_FSL.connect(
        T1_to_dwi_flt, 'out_matrix_file', T1_to_dwi_applyxfm, 'in_matrix_file')
    frontal_connectivity_FSL.connect(
        fslroi, 'roi_file', MNI_to_dwi_flt, 'reference')
    frontal_connectivity_FSL.connect(
        frontalROI, 'frontalROI', MNI_to_dwi_applyxfm_seed, 'in_file')
    frontal_connectivity_FSL.connect(
        fslroi, 'roi_file', MNI_to_dwi_applyxfm_seed, 'reference')
    frontal_connectivity_FSL.connect(
        MNI_to_dwi_flt, 'out_matrix_file',
        MNI_to_dwi_applyxfm_seed, 'in_matrix_file')
    frontal_connectivity_FSL.connect(
        MNI_to_dwi_applyxfm_seed, 'out_file',
        MNI_to_dwi_threshold_seed, 'in_file')
    frontal_connectivity_FSL.connect(
        targetROI, 'targetROI', MNI_to_dwi_applyxfm_target, 'in_file')
    frontal_connectivity_FSL.connect(
        fslroi, 'roi_file', MNI_to_dwi_applyxfm_target, 'reference')
    frontal_connectivity_FSL.connect(
        MNI_to_dwi_flt, 'out_matrix_file',
        MNI_to_dwi_applyxfm_target, 'in_matrix_file')
    frontal_connectivity_FSL.connect(
        MNI_to_dwi_applyxfm_target, 'out_file',
        MNI_to_dwi_threshold_target, 'in_file')
    frontal_connectivity_FSL.connect(
        T1_to_dwi_applyxfm, 'out_file', multiply, 'in_file')
    frontal_connectivity_FSL.connect(
        MNI_to_dwi_threshold_seed, 'out_file', multiply, 'operand_file')
    frontal_connectivity_FSL.connect(
        T1_to_dwi_applyxfm, 'out_file', multiply2, 'in_file')
    frontal_connectivity_FSL.connect(
        MNI_to_dwi_threshold_target, 'out_file', multiply2, 'operand_file')

    # DWI processing
    frontal_connectivity_FSL.connect(selectfiles, 'dwi', denoise, 'in_file')
    frontal_connectivity_FSL.connect(denoise, 'out_file', eddycorrect, 'in_file')
    frontal_connectivity_FSL.connect(eddycorrect, 'eddy_corrected', fslroi, 'in_file')
    frontal_connectivity_FSL.connect(fslroi, 'roi_file', dwi_bet, 'in_file')
    frontal_connectivity_FSL.connect(
        eddycorrect, 'eddy_corrected', dti, 'dwi')
    frontal_connectivity_FSL.connect(dwi_bet, 'mask_file', dti, 'mask')
    frontal_connectivity_FSL.connect(selectfiles, 'bvec', dti, 'bvecs')
    frontal_connectivity_FSL.connect(selectfiles, 'bval', dti, 'bvals')

    frontal_connectivity_FSL.connect(
        eddycorrect, 'eddy_corrected', bedp, 'dwi')
    frontal_connectivity_FSL.connect(dwi_bet, 'mask_file', bedp, 'mask')
    frontal_connectivity_FSL.connect(selectfiles, 'bvec', bedp, 'bvecs')
    frontal_connectivity_FSL.connect(selectfiles, 'bval', bedp, 'bvals')

    frontal_connectivity_FSL.connect(dwi_bet, 'mask_file', pbx, 'mask')
    frontal_connectivity_FSL.connect(bedp, 'mean_phsamples', pbx, 'phsamples')
    frontal_connectivity_FSL.connect(bedp, 'mean_thsamples', pbx, 'thsamples')
    frontal_connectivity_FSL.connect(bedp, 'mean_fsamples', pbx, 'fsamples')

    # Moving density image to common space
    frontal_connectivity_FSL.connect(
        MNI_to_dwi_flt, 'out_matrix_file', invt, 'in_file')
    frontal_connectivity_FSL.connect(
        invt, 'out_file', dwi_to_MNI_applyxfm, 'in_matrix_file')

    # Running the workflow
    frontal_connectivity_FSL.base_dir = os.path.abspath(out_directory)
    frontal_connectivity_FSL.write_graph()
    frontal_connectivity_FSL.run(plugin='PBSGraph')

import pandas as pd
base_directory = '/imaging/jb07/CALM/CALM_BIDS/'
out_directory = '/imaging/jb07/CALM/Behavioural/Community_Detection/'
df = pd.read_csv('/imaging/jb07/CALM/Displacements_Results.csv')
subject_list = df[df['maximum displacement [mm]'] < 3]['Unnamed: 0'].values
frontal_connectivity_FSL([subject_list[0]], base_directory, out_directory)
