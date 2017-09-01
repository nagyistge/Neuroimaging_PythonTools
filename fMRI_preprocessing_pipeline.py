def fMRI_preprocess(folder,participant):
    
    # Creating folder structure 
    import os
    
    if os.path.isdir(folder + 'float_convert/') == False:
      os.makedirs(folder + 'float_convert/')
        
    if os.path.isdir(folder + 'time_corrected/') == False:
      os.makedirs(folder + 'time_corrected/')
            
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
    
    if os.path.isdir(folder + 'coregistered_image/') == False:
      os.makedirs(folder + 'coregistered_image/')

    if os.path.isdir(folder + 'smoothed_image/') == False:
      os.makedirs(folder + 'smoothed_image/')
    
    if os.path.isfile('/imaging/jb07/FreeSurfer/' + participant + '/mri/T1.nii.gz') == False:
        print 'Converting Freesurfer T1'
        # Converting FreeSurfer brain from .mgz to .nii.gz
        import nipype.interfaces.io as nio 
        fs_source = nio.FreeSurferSource()
        fs_source.inputs.subject_id = participant
        fs_source.inputs.subjects_dir = '/imaging/jb07/FreeSurfer/'
        fs_files = fs_source.run()
        
        import nipype.interfaces.freesurfer as fs
        mri_convert = fs.MRIConvert()
        mri_convert.inputs.in_file = fs_files.outputs.brain
        mri_convert.inputs.out_type = 'niigz'
        mri_convert.inputs.out_file = '/imaging/jb07/FreeSurfer/' + participant + '/mri/T1.nii.gz'
        mri_convert.run() 

    # Convert to float representation
    import nipype.interfaces.fsl as fsl
    print 'Converting to float representation'
    maths = fsl.ImageMaths(out_data_type='float',op_string = '',suffix='_dtype')
    maths.inputs.in_file = folder + 'raw_data/' + participant + '.nii.gz'
    maths.inputs.out_file = folder + 'float_convert/' + participant + '.nii.gz'
    maths.run()
    
    # Slice time correction
    print 'Slice time correction'
    st = fsl.SliceTimer()
    st.inputs.in_file = folder + 'float_convert/' + participant + '.nii.gz'
    st.inputs.time_repetition = 2
    st.inputs.interleaved = True
    st.inputs.out_file = folder + 'time_corrected/' + participant + '.nii.gz'
    st.run()

    # Extract the middle volume for re-alignment
    import nibabel as nib
    import numpy as np
    print 'Extracting middle volume'
    img = nib.load(folder + 'time_corrected/' + participant + '.nii.gz')
    data = img.get_data()
    affine = img.get_affine()
    dimensions = np.shape(data)
    middle_volume = data[:,:,:,dimensions[3]/2]
    middle_volume = nib.Nifti1Image(middle_volume, affine)
    nib.save(middle_volume,folder + 'reference_volume/' + participant + '.nii.gz')
    
    # Motion correction through co-registration to middle volume
    print 'Motion correction'
    os.chdir(folder + 'motion_corrected/')
    mcflirt =fsl.MCFLIRT(save_mats = True,save_plots = True)
    mcflirt.inputs.in_file = folder + 'float_convert/' + participant + '.nii.gz'
    mcflirt.inputs.ref_file = folder + 'reference_volume/' + participant + '.nii.gz'
    mcflirt.inputs.out_file = folder + 'motion_corrected/' + participant + '.nii.gz'
    mcflirt.run()
    
    # Plotting motion parameters
    movement_types = ['rotations','translations','displacement']
    
    for movement_type in movement_types:
        plot_motion = fsl.PlotMotionParams(in_source='fsl')
        plot_motion.inputs.plot_type = movement_type
        plot_motion.inputs.in_file = folder + 'motion_corrected/' + participant + '.nii.gz.par'
        plot_motion.inputs.out_file = folder + 'motion_corrected/' + participant + '_' + movement_type + '.png'
        plot_motion.run()
    
    # Calculating a mean image 
    mean_image = fsl.ImageMaths(op_string = '-Tmean', suffix='_mean')
    mean_image.inputs.in_file = folder + 'motion_corrected/' + participant + '.nii.gz'
    mean_image.inputs.out_file = folder + 'mean_image/' + participant + '.nii.gz'
    mean_image.run()
    
    # Creating a brain mask based on the mean image
    print 'Calculating brain mask'
    skullstrip = interface=fsl.BET(mask = True, no_output=True, frac = 0.3)
    skullstrip.inputs.in_file = folder + 'mean_image/' + participant + '.nii.gz'
    skullstrip.inputs.out_file = folder + 'brain_mask/prelim_' + participant + '.nii.gz'
    skullstrip.run()
    
    # Mask the functional image
    mask_image = fsl.ImageMaths(op_string='-mas')
    mask_image.inputs.in_file = folder + 'mean_image/' + participant + '.nii.gz'
    mask_image.inputs.in_file2 = folder + 'brain_mask/prelim_' + participant + '_mask.nii.gz'
    mask_image.inputs.out_file = folder + 'masked_image/prelim_' + participant + '.nii.gz'
    mask_image.run()
    
    # Co-register functional to structural image
    from nipype.interfaces import fsl
    flt = fsl.FLIRT(dof=6, cost_func='corratio')
    flt.inputs.in_file = folder + 'masked_image/prelim_' + participant + '.nii.gz'
    flt.inputs.reference = '/imaging/jb07/FreeSurfer/' + participant + '/mri/T1.nii.gz'
    flt.inputs.out_file = folder + 'coregistered_image/middle_' + participant + '.nii.gz'
    flt.inputs.out_matrix_file = folder + 'coregistered_image/middle_' + participant + '.mat'
    flt.run() 
    
    applyxfm = fsl.ApplyXfm()
    applyxfm.inputs.in_file = folder + 'motion_corrected/' + participant + '.nii.gz'
    applyxfm.inputs.in_matrix_file = folder + 'coregistered_image/middle_' + participant + '.mat'
    applyxfm.inputs.out_file = folder + 'coregistered_image/' + participant + '.nii.gz'
    applyxfm.inputs.reference = folder + 'masked_image/prelim_' + participant + '.nii.gz'
    applyxfm.inputs.apply_xfm = True
    result = applyxfm.run() 

    """
    file_list = list()

    functional_img = nib.load(folder + 'motion_corrected/' + participant + '.nii.gz')
    functional_data = functional_img.get_data()
    functional_affine = functional_img.get_affine()
    
    structural_img = nib.load('/imaging/jb07/FreeSurfer/' + participant + '/mri/T1.nii.gz')
    structural_affine = structural_img.get_affine()
    
    number_of_images = np.shape(functional_data)
    number_of_images = number_of_images[3]
    
    for image in range(0,number_of_images):
        single_image = functional_data[:,:,:,image]
        filename = folder + 'motion_corrected/' + participant + '_temp.nii.gz'
        nib.save(nib.Nifti1Image(single_image, functional_affine), filename)
        
        flt = fsl.FLIRT(dof=6)
        flt.inputs.in_file = filename
        flt.inputs.reference = '/imaging/jb07/FreeSurfer/' + participant + '/mri/T1.nii.gz'
        flt.inputs.out_file = folder + 'coregistered_image/' + participant + '_temp_' + str(image+1) + '.nii.gz'
        flt.inputs.in_matrix_file = folder + 'coregistered_image/middle_' + participant + '.mat'
        flt.inputs.apply_xfm = True
        flt.run() 
        file_list.append(folder + 'coregistered_image/' + participant + '_temp_' + str(image+1) + '.nii.gz')
    

    # Merging volumes to a single image    
    from nipype.interfaces.fsl import Merge
    merger = Merge()
    merger.inputs.in_files = file_list
    merger.inputs.dimension = 't'
    merger.inputs.output_type = 'NIFTI_GZ'
    merger.inputs.merged_file = folder + 'coregistered_image/' + participant + '.nii.gz'
    merger.run()
    
    for afile in file_list:
        os.remove(afile)    
    """
    
    # Mask the functional image
    print 'Calculating brain mask 2'
    skullstrip = interface=fsl.BET(mask = True, no_output=True, frac = 0.3)
    skullstrip.inputs.in_file = folder + 'coregistered_image/middle_' + participant + '.nii.gz'
    skullstrip.inputs.out_file = folder + 'brain_mask/' + participant + '.nii.gz'
    skullstrip.run()
    
    mask_image = fsl.ImageMaths(op_string='-mas')
    mask_image.inputs.in_file = folder + 'coregistered_image/' + participant + '.nii.gz'
    mask_image.inputs.in_file2 = folder + 'brain_mask/' + participant + '_mask.nii.gz'
    mask_image.inputs.out_file = folder + 'masked_image/' + participant + '.nii.gz'
    mask_image.run()

    # Getting the median value
    medianval = fsl.ImageStats(op_string='-k %s -p 50')
    medianval.inputs.in_file = folder + 'coregistered_image/' + participant + '.nii.gz'
    medianval.inputs.mask_file = folder + 'brain_mask/' + participant + '_mask.nii.gz'
    medianval = medianval.run()
    
    # Smoothing
    print 'Smoothing'
    smooth = fsl.SUSAN()
    smooth.inputs.in_file = folder + 'coregistered_image/' + participant + '.nii.gz'
    smooth.inputs.brightness_threshold = 0.75*medianval.outputs.out_stat
    smooth.inputs.fwhm = 5
    smooth.inputs.out_file = folder + 'smoothed_image/' + participant + '.nii.gz'
    smooth.run()
