def fMRI_preprocessing(folder,participant):
    
    # Creating folder structure 
    import os
        
    if os.path.isdir(folder + 'float_convert/') == False:
      os.makedirs(folder + 'float_convert/')

    if os.path.isdir(folder + 'reference_volume/') == False:
      os.makedirs(folder + 'reference_volume/')

    if os.path.isdir(folder + 'motion_corrected/') == False:
      os.makedirs(folder + 'motion_corrected/')
        
    if os.path.isdir(folder + 'mean_image/') == False:
      os.makedirs(folder + 'mean_image/')
        
    if os.path.isdir(folder + 'brain_mask/') == False:
      os.makedirs(folder + 'brain_mask/')
    
    if os.path.isdir(folder + 'masked_image/') == False:
      os.makedirs(folder + 'masked_image/')
        
    if os.path.isdir(folder + 'smoothed_image/') == False:
      os.makedirs(folder + 'smoothed_image/')
             
    if os.path.isdir(folder + 'highpass_filtered/') == False:
      os.makedirs(folder + 'highpass_filtered/')

    if os.path.isdir(folder + 'artefact_detection/') == False:
      os.makedirs(folder + 'artefact_detection/')
    
    # Convert to float representation
    print 'Converting to float representation'
    import nipype.interfaces.fsl as fsl
    maths = fsl.ImageMaths(out_data_type='float',op_string = '',suffix='_dtype')
    maths.inputs.in_file = folder + 'raw_data/' + participant + '.nii.gz'
    maths.inputs.out_file = folder + 'float_convert/' + participant + '.nii.gz'
    maths.run()
    
    # Extract the middle volume for re-alignment
    print 'Extracting middle volume'
    import nibabel as nib
    import numpy as np
    img = nib.load(folder + 'float_convert/' + participant + '.nii.gz')
    data = img.get_data()
    affine = img.get_affine()
    dimensions = np.shape(data)
    middle_volume = data[:,:,:,dimensions[3]/2]
    
    middle_volume = nib.Nifti1Image(middle_volume, affine)
    nib.save(middle_volume,folder + 'reference_volume/' + participant + '.nii.gz')
    
    # Motion correction through co-registration to middle volume
    print 'Running motion correction'
    os.chdir(folder + 'motion_corrected/')
    mcflirt =fsl.MCFLIRT(save_mats = True,save_plots = True)
    mcflirt.inputs.in_file = folder + 'float_convert/' + participant + '.nii.gz'
    mcflirt.inputs.ref_file = folder + 'reference_volume/' + participant + '.nii.gz'
    mcflirt.inputs.out_file = folder + 'motion_corrected/' + participant + '.nii.gz'
    mcflirt.run()
    
    # Plotting motion parameters
    print 'Plotting motion parameters'
    movement_types = ['rotations','translations','displacement']
    
    for movement_type in movement_types:
        plot_motion = fsl.PlotMotionParams(in_source='fsl')
        plot_motion.inputs.plot_type = movement_type
        plot_motion.inputs.in_file = folder + 'motion_corrected/' + participant + '.nii.gz.par'
        plot_motion.inputs.out_file = folder + 'motion_corrected/' + participant + '_' + movement_type + '.png'
        plot_motion.run()
        
    # Calculating a mean image 
    print 'Calculating mean image'
    mean_image = fsl.ImageMaths(op_string = '-Tmean', suffix='_mean')
    mean_image.inputs.in_file = folder + 'motion_corrected/' + participant + '.nii.gz'
    mean_image.inputs.out_file = folder + 'mean_image/' + participant + '.nii.gz'
    mean_image.run()
    
    # Creating a brain mask based on the mean image
    print 'Creating brain mask'
    skullstrip = interface=fsl.BET(mask = True, no_output=True, frac = 0.3)
    skullstrip.inputs.in_file = folder + 'mean_image/' + participant + '.nii.gz'
    skullstrip.inputs.out_file = folder + 'brain_mask/' + participant + '.nii.gz'
    skullstrip.run()
    
    # Mask the functional image
    print 'Masking functional image'
    mask_image = fsl.ImageMaths(op_string='-mas')
    mask_image.inputs.in_file = folder + 'mean_image/' + participant + '.nii.gz'
    mask_image.inputs.in_file2 = folder + 'brain_mask/' + participant + '_mask.nii.gz'
    mask_image.inputs.out_file = folder + 'masked_image/' + participant + '.nii.gz'
    mask_image.run()
    
    # Getting the median value
    medianval = fsl.ImageStats(op_string='-k %s -p 50')
    medianval.inputs.in_file = folder + 'motion_corrected/' + participant + '.nii.gz'
    medianval.inputs.mask_file = folder + 'brain_mask/' + participant + '_mask.nii.gz'
    medianval = medianval.run()
    
    # Smoothing
    print 'Smoothing image'
    smooth = fsl.SUSAN()
    smooth.inputs.in_file = folder + 'masked_image/' + participant + '.nii.gz'
    smooth.inputs.brightness_threshold = 0.75*medianval.outputs.out_stat
    smooth.inputs.fwhm = 10
    smooth.inputs.out_file = folder + 'smoothed_image/' + participant + '.nii.gz'
    smooth.run()
    
    # Highpass filtering
    print 'High-pass filtering'
    highpass = fsl.ImageMaths()
    highpass.inputs.in_file = folder + 'smoothed_image/' + participant + '.nii.gz'
    highpass.inputs.out_file = folder + 'highpass_filtered/' + participant + '.nii.gz'
    highpass.inputs.op_string = '-bptf 120/2'
    highpass.run()
    
    # Automatic artefact detection
    import nipype.algorithms.rapidart as ra
    os.chdir(folder + 'artefact_detection/')
    
    art = ra.ArtifactDetect(use_norm = True, norm_threshold = 1, zintensity_threshold = 3, parameter_source = 'FSL')
    art.inputs.realigned_files = folder + 'motion_corrected/' + participant + '.nii.gz'
    art.inputs.realignment_parameters = folder + 'motion_corrected/' + participant + '.nii.gz.par'
    art.inputs.mask_file = folder + 'brain_mask/' + participant + '_mask.nii.gz'
    art.inputs.mask_type = 'file'
    art.run()

    if __name__ == '__main__':
      main()