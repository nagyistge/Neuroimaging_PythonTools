def frontal_connectivity(subject_list, base_directory, out_directory):

    # Loading required packages
    import nibabel as nib
    from nipype.interfaces.ants import N4BiasFieldCorrection
    import nipype.interfaces.fsl as fsl
    import nipype.interfaces.mrtrix as mrt
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

    img = nib.load(atlas_file)
    atlas_data = img.get_data()
    atlas_affine = img.get_affine()
    atlas_data[atlas_data > 6] = 0
    atlas_data[atlas_data == 2] = 0
    atlas_data[atlas_data > 0] = 1
    nib.save(nib.Nifti1Image(atlas_data, atlas_affine),
             out_directory + '/frontal_connectivity/frontalROIs.nii.gz')

    img = nib.load(atlas_file)
    atlas_data = img.get_data()
    atlas_data[atlas_data == 2] = 99
    atlas_data[atlas_data < 6] = 0
    atlas_data[atlas_data > 0] = 1
    number_of_voxels = len(atlas_data[atlas_data == 1])
    nib.save(nib.Nifti1Image(atlas_data, atlas_affine),
             out_directory + '/frontal_connectivity/targetROIs.nii.gz')

    frontalROI = pe.Node(interface=util.Function(
        input_names=['out_directory'],
        output_names=['frontalROI']),
        name='frontalROI')

    frontalROI.inputs.function_str = "def pass_filename(out_directory):   return out_directory + '/frontal_connectivity/frontalROIs.nii.gz'"
    frontalROI.inputs.out_directory = out_directory

    targetROI = pe.Node(interface=util.Function(
        input_names=['out_directory'],
        output_names=['targetROI']),
        name='targetROI')

    targetROI.inputs.function_str = "def pass_filename(out_directory):    return out_directory + '/frontal_connectivity/targetROIs.nii.gz'"
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
    dwi2tensor = pe.Node(interface=mrt.DWI2Tensor(), name='dwi2tensor')
    tensor2vector = pe.Node(
        interface=mrt.Tensor2Vector(), name='tensor2vector')
    tensor2adc = pe.Node(
        interface=mrt.Tensor2ApparentDiffusion(), name='tensor2adc')
    tensor2fa = pe.Node(
        interface=mrt.Tensor2FractionalAnisotropy(), name='tensor2fa')

    # Fitting the CSD model
    fsl2mrtrix = pe.Node(interface=mrt.FSL2MRTrix(
        invert_x=True), name='fsl2mrtrix')
    gunzip = pe.Node(interface=misc.Gunzip(), name='gunzip')
    gunzip2 = pe.Node(interface=misc.Gunzip(), name='gunzip2')
    gunzip3 = pe.Node(interface=misc.Gunzip(), name='gunzip3')
    gunzip4 = pe.Node(interface=misc.Gunzip(), name='gunzip4')

    erode_mask_firstpass = pe.Node(
        interface=mrt.Erode(), name='erode_mask_firstpass')
    erode_mask_secondpass = pe.Node(
        interface=mrt.Erode(), name='erode_mask_secondpass')
    MRmultiply = pe.Node(interface=mrt.MRMultiply(), name='MRmultiply')
    MRmult_merge = pe.Node(interface=util.Merge(2), name='MRmultiply_merge')
    threshold_FA = pe.Node(interface=mrt.Threshold(
        absolute_threshold_value=0.7), name='threshold_FA')

    # White matter mask
    gen_WM_mask = pe.Node(
        interface=mrt.GenerateWhiteMatterMask(), name='gen_WM_mask')
    threshold_wmmask = pe.Node(interface=mrt.Threshold(
        absolute_threshold_value=0.4), name='threshold_wmmask')

    # CSD probabilistic tractography
    estimateresponse = pe.Node(interface=mrt.EstimateResponseForSH(
        maximum_harmonic_order=8), name='estimateresponse')
    csdeconv = pe.Node(interface=mrt.ConstrainedSphericalDeconvolution(
        maximum_harmonic_order=8), name='csdeconv')

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

    # Tracking
    probCSDstreamtrack = pe.Node(
        interface=mrt.ProbabilisticSphericallyDeconvolutedStreamlineTrack(),
        name='probCSDstreamtrack')
    probCSDstreamtrack.inputs.inputmodel = 'SD_PROB'
    probCSDstreamtrack.inputs.desired_number_of_tracks = number_of_voxels*100
    tck2trk = pe.Node(interface=mrt.MRTrix2TrackVis(), name='tck2trk')

    # Converting to a density image
    track_density = pe.Node(interface=mrt.Tracks2Prob(
        args='-lstdi'), name='track_density')

    mrconvert = pe.Node(interface=mrt.MRConvert(
        extension='nii'), name='mrconvert')

    # Moving the density image to common space
    invt = pe.Node(interface=fsl.ConvertXFM(invert_xfm=True), name='invt')
    dwi_to_MNI_applyxfm = pe.Node(fsl.ApplyXfm(
        apply_xfm=True), name='dwi_to_MNI_applyxfm')
    dwi_to_MNI_applyxfm.inputs.reference = os.environ[
        'FSLDIR'] + '/data/standard/MNI152_T1_1mm_brain.nii.gz'

    # Setting up the workflow
    frontal_connectivity = pe.Workflow(name='frontal_connectivity')

    # T1 processing
    frontal_connectivity.connect(
        infosource, 'subject_id', selectfiles, 'subject_id')
    frontal_connectivity.connect(selectfiles, 'T1', robustfov, 'in_file')
    frontal_connectivity.connect(robustfov, 'out_roi', T1_bet, 'in_file')
    frontal_connectivity.connect(T1_bet, 'out_file', n4, 'input_image')
    frontal_connectivity.connect(n4, 'output_image', fast, 'in_files')
    frontal_connectivity.connect(
        fast, 'partial_volume_files', select1, 'inlist')
    frontal_connectivity.connect(
        fast, 'partial_volume_files', select2, 'inlist')
    frontal_connectivity.connect(select1, 'out', overlap, 'in_file1')
    frontal_connectivity.connect(select2, 'out', overlap, 'in_file2')

    # Getting the ROI into dwi space
    frontal_connectivity.connect(n4, 'output_image', T1_to_dwi_flt, 'in_file')
    frontal_connectivity.connect(
        fslroi, 'roi_file', T1_to_dwi_flt, 'reference')
    frontal_connectivity.connect(
        overlap, 'out_file', T1_to_dwi_applyxfm, 'in_file')
    frontal_connectivity.connect(
        fslroi, 'roi_file', T1_to_dwi_applyxfm, 'reference')
    frontal_connectivity.connect(
        T1_to_dwi_flt, 'out_matrix_file', T1_to_dwi_applyxfm, 'in_matrix_file')
    frontal_connectivity.connect(
        fslroi, 'roi_file', MNI_to_dwi_flt, 'reference')
    frontal_connectivity.connect(
        frontalROI, 'frontalROI', MNI_to_dwi_applyxfm_seed, 'in_file')
    frontal_connectivity.connect(
        fslroi, 'roi_file', MNI_to_dwi_applyxfm_seed, 'reference')
    frontal_connectivity.connect(
        MNI_to_dwi_flt, 'out_matrix_file',
        MNI_to_dwi_applyxfm_seed, 'in_matrix_file')
    frontal_connectivity.connect(
        MNI_to_dwi_applyxfm_seed, 'out_file',
        MNI_to_dwi_threshold_seed, 'in_file')
    frontal_connectivity.connect(
        targetROI, 'targetROI', MNI_to_dwi_applyxfm_target, 'in_file')
    frontal_connectivity.connect(
        fslroi, 'roi_file', MNI_to_dwi_applyxfm_target, 'reference')
    frontal_connectivity.connect(
        MNI_to_dwi_flt, 'out_matrix_file',
        MNI_to_dwi_applyxfm_target, 'in_matrix_file')
    frontal_connectivity.connect(
        MNI_to_dwi_applyxfm_target, 'out_file',
        MNI_to_dwi_threshold_target, 'in_file')
    frontal_connectivity.connect(
        T1_to_dwi_applyxfm, 'out_file', multiply, 'in_file')
    frontal_connectivity.connect(
        MNI_to_dwi_threshold_seed, 'out_file', multiply, 'operand_file')
    frontal_connectivity.connect(
        T1_to_dwi_applyxfm, 'out_file', multiply2, 'in_file')
    frontal_connectivity.connect(
        MNI_to_dwi_threshold_target, 'out_file', multiply2, 'operand_file')

    # DWI processing
    frontal_connectivity.connect(selectfiles, 'dwi', denoise, 'in_file')
    frontal_connectivity.connect(denoise, 'out_file', eddycorrect, 'in_file')
    frontal_connectivity.connect(
        eddycorrect, 'eddy_corrected', fslroi, 'in_file')
    frontal_connectivity.connect(fslroi, 'roi_file', dwi_bet, 'in_file')
    frontal_connectivity.connect(
        eddycorrect, 'eddy_corrected', gunzip, 'in_file')
    frontal_connectivity.connect(dwi_bet, 'mask_file', gunzip2, 'in_file')

    frontal_connectivity.connect(selectfiles, 'bval', fsl2mrtrix, 'bval_file')
    frontal_connectivity.connect(selectfiles, 'bvec', fsl2mrtrix, 'bvec_file')
    frontal_connectivity.connect(gunzip, 'out_file', dwi2tensor, 'in_file')
    frontal_connectivity.connect(
        fsl2mrtrix, 'encoding_file', dwi2tensor, 'encoding_file')
    frontal_connectivity.connect(
        dwi2tensor, 'tensor', tensor2vector, 'in_file')
    frontal_connectivity.connect(dwi2tensor, 'tensor', tensor2adc, 'in_file')
    frontal_connectivity.connect(dwi2tensor, 'tensor', tensor2fa, 'in_file')
    frontal_connectivity.connect(tensor2fa, 'FA', MRmult_merge, 'in1')

    # Thresholding to create a mask of single fibre voxels
    frontal_connectivity.connect(
        gunzip2, 'out_file', erode_mask_firstpass, 'in_file')
    frontal_connectivity.connect(
        erode_mask_firstpass, 'out_file', erode_mask_secondpass, 'in_file')
    frontal_connectivity.connect(
        erode_mask_secondpass, 'out_file', MRmult_merge, 'in2')
    frontal_connectivity.connect(MRmult_merge, 'out', MRmultiply, 'in_files')
    frontal_connectivity.connect(
        MRmultiply, 'out_file', threshold_FA, 'in_file')
    frontal_connectivity.connect(gunzip, 'out_file', gen_WM_mask, 'in_file')
    frontal_connectivity.connect(
        gunzip2, 'out_file', gen_WM_mask, 'binary_mask')
    frontal_connectivity.connect(
        fsl2mrtrix, 'encoding_file', gen_WM_mask, 'encoding_file')
    frontal_connectivity.connect(
        gen_WM_mask, 'WMprobabilitymap', threshold_wmmask, 'in_file')

    # Estimate response
    frontal_connectivity.connect(
        gunzip, 'out_file', estimateresponse, 'in_file')
    frontal_connectivity.connect(
        fsl2mrtrix, 'encoding_file', estimateresponse, 'encoding_file')
    frontal_connectivity.connect(
        threshold_FA, 'out_file', estimateresponse, 'mask_image')

    # CSD calculation
    frontal_connectivity.connect(gunzip, 'out_file', csdeconv, 'in_file')
    frontal_connectivity.connect(
        gen_WM_mask, 'WMprobabilitymap', csdeconv, 'mask_image')
    frontal_connectivity.connect(
        estimateresponse, 'response', csdeconv, 'response_file')
    frontal_connectivity.connect(
        fsl2mrtrix, 'encoding_file', csdeconv, 'encoding_file')

    # Running the tractography
    frontal_connectivity.connect(multiply, "out_file", gunzip3, 'in_file')
    frontal_connectivity.connect(multiply2, "out_file", gunzip4, 'in_file')

    frontal_connectivity.connect(
        gunzip3, 'out_file', probCSDstreamtrack, "seed_file")
    frontal_connectivity.connect(
        gunzip4, 'out_file', probCSDstreamtrack, "include_file")
    frontal_connectivity.connect(
        csdeconv, "spherical_harmonics_image", probCSDstreamtrack, "in_file")
    frontal_connectivity.connect(
        gunzip2, "out_file", probCSDstreamtrack, "mask_file")
    frontal_connectivity.connect(gunzip, "out_file", tck2trk, "image_file")
    frontal_connectivity.connect(
        probCSDstreamtrack, "tracked", tck2trk, "in_file")

    # Creating a track density image
    frontal_connectivity.connect(
        probCSDstreamtrack, "tracked", track_density, 'in_file')
    frontal_connectivity.connect(
        tensor2fa, 'FA', track_density, 'template_file')
    frontal_connectivity.connect(
        track_density, 'tract_image', mrconvert, 'in_file')

    # Moving density image to common space
    frontal_connectivity.connect(
        MNI_to_dwi_flt, 'out_matrix_file', invt, 'in_file')
    frontal_connectivity.connect(
        invt, 'out_file', dwi_to_MNI_applyxfm, 'in_matrix_file')
    frontal_connectivity.connect(
        mrconvert, 'converted', dwi_to_MNI_applyxfm, 'in_file')

    # Running the workflow
    frontal_connectivity.base_dir = os.path.abspath(out_directory)
    frontal_connectivity.write_graph()
    frontal_connectivity.run(plugin='PBSGraph')

import pandas as pd
base_directory = '/imaging/jb07/CALM/CALM_BIDS/'
out_directory = '/imaging/jb07/CALM/Behavioural/Community_Detection/'
df = pd.read_csv('/imaging/jb07/CALM/Displacements_Results.csv')
subject_list = df[df['maximum displacement [mm]'] < 3]['Unnamed: 0'].values
frontal_connectivity([subject_list[0]], base_directory, out_directory)
