def unzip_image_file(image_file):
    import os
    import nibabel as nib
    img = nib.load(image_file)
    data = img.get_data()
    affine = img.get_affine()
    
    nib.save(nib.Nifti1Image(data,affine),image_file[:-3])
    os.remove(image_file)

def denoise_dwi(infile,outfile):
    import nibabel as nib
    import numpy as np
    import matplotlib.pyplot as plt
    from dipy.denoise.nlmeans import nlmeans
    
    img = nib.load(infile)
    data = img.get_data()
    affine = img.get_affine()
    mask = data[..., 0] > 80
    a = data.shape 
    
    denoised_data = np.ndarray(shape=data.shape)
    for image in range(0,a[3]):
        print(str(image + 1) + '/' + str(a[3] + 1))
        dat = data[...,image]
        sigma = np.std(dat[~mask]) # Calculating the standard deviation of the noise 
        den = nlmeans(dat, sigma=sigma, mask=mask)
        denoised_data[:,:,:,image] = den
        
    nib.save(nib.Nifti1Image(denoised_data, affine), outfile)

def eddy_correction(infile,outfile):
    from subprocess import call
    command = 'eddy_correct ' + infile + ' ' + outfile + ' ' + str(1)
    call(command,shell=True)

def upsample_dwi(infile,outfile): 
    import nibabel as nib
    import numpy as np
    from dipy.align.reslice import reslice    
    infile
    img = nib.load(infile)
    affine = img.get_affine()
    data = img.get_data()
    new_array = np.ndarray(shape=(192,192,136,69))
    
    for i in range(0,69):
        image = data[...,i]
        zooms = img.get_header().get_zooms()[:3]
        new_zooms = (1., 1., 1.)
        data2, affine2 = reslice(image, affine, zooms, new_zooms)
        new_array[...,i] = data2
    nib.save(nib.Nifti1Image(new_array, affine2), outfile)

def extract_b0(infile,outfile):
    import nibabel as nib
    dwi_image = infile
    img = nib.load(dwi_image)
    data = img.get_data()
    affine = img.get_affine()
    
    b0 = data[...,0]
    nib.save(nib.Nifti1Image(b0, affine), outfile)

def MNI_coregister(b0,infile,outfile):
    
   # 1. Rigid co-registration between b0 image and MNI image
    from nipype.interfaces import fsl
    flt = fsl.FLIRT(dof=6, cost_func='corratio')
    flt.inputs.in_file = b0
    flt.inputs.reference = '/imaging/local/linux/bin/fsl-4.1.2/data/standard/MNI152_T1_1mm.nii.gz'
    flt.inputs.out_file = outfile[:-7] + '_b0_coreg.nii.gz'
    flt.inputs.out_matrix_file = outfile[:-7] + '_b0_coreg.mat'
    flt.run() 
    
    # 2. Applying co-registration to all diffusion images
    # Co-register functional to structural image
    import nibabel as nib
    import numpy as np
    from IPython.display import clear_output
    
    infile = infile[:-7] + '.nii.gz'
    diffusion_image = nib.load(infile)
    diffusion_data = diffusion_image.get_data()
    diffusion_affine = diffusion_image.get_affine()
    
    b0_img = nib.load(outfile[:-7] + '_b0_coreg.nii.gz')
    b0_affine = b0_img.get_affine()
    
    number_of_images = np.shape(diffusion_data)
    number_of_images = number_of_images[3]
    
    import os
    new_array = np.empty(shape=[182,218,182,69])
    
    for image in range(0,number_of_images):
        print image
        single_image = diffusion_data[...,image]
        filename = outfile[:-7] + '_temp.nii.gz'
        nib.save(nib.Nifti1Image(single_image, diffusion_affine), filename)
        
        import nipype.interfaces.fsl as fsl
        applyxfm = fsl.ApplyXfm()
        applyxfm.inputs.in_file = filename
        applyxfm.inputs.in_matrix_file =  outfile[:-7] + '_b0_coreg.mat'
        applyxfm.inputs.out_file = outfile[:-7] + '_temp_' + str(image+1) + '.nii.gz'
        applyxfm.inputs.reference = outfile[:-7] + '_b0_coreg.nii.gz'
        applyxfm.inputs.apply_xfm = True
        result = applyxfm.run() 
        
        temp_img = nib.load(outfile[:-7] + '_temp_' + str(image+1) + '.nii.gz')
        temp_data = temp_img.get_data()
        new_array[...,image] = temp_data
        
        os.remove(outfile[:-7] + '_temp_' + str(image+1) + '.nii.gz')
        clear_output()
        
    nib.save(nib.Nifti1Image(new_array, b0_affine), outfile)
    os.remove(filename)

def create_brain_mask(infile,outfile):
    import os
    path = os.path.split(outfile)
    os.chdir(path[0])
    
    from nipype.interfaces import fsl
    btr = fsl.BET()
    btr.inputs.in_file = infile
    btr.inputs.frac = 0.3
    btr.inputs.robust = False
    btr.inputs.no_output = True
    btr.inputs.mask = True
    btr.run()

def mask_dwi(infile,maskfile,outfile):
    import nibabel as nib
    import numpy as np
    img = nib.load(infile)
    dwi_data = img.get_data()
    dwi_affine = img.get_affine()
    
    img = nib.load(maskfile)
    mask_data = img.get_data()
    mask_affine = img.get_affine()
    
    mask_data = np.repeat(mask_data,repeats=69,axis=2)
    mask_data = np.reshape(mask_data, newshape=[182,218,182,69])
    dwi_data = dwi_data * mask_data
    
    nib.save(nib.Nifti1Image(dwi_data, dwi_affine), outfile)

def fit_diffusion_tensor(infile,encoding_file,outfile):
    import nipype.interfaces.mrtrix as mrt
    dwi2tensor = mrt.DWI2Tensor()
    dwi2tensor.inputs.in_file = infile[:-3]
    dwi2tensor.inputs.encoding_file = encoding_file
    dwi2tensor.inputs.out_filename = outfile
    dwi2tensor.run()     

def calculate_tensor_metric(infile,FA_outfile,EV_outfile):
    import nipype.interfaces.mrtrix as mrt
    tensor2FA = mrt.Tensor2FractionalAnisotropy()
    tensor2FA.inputs.in_file = infile
    tensor2FA.inputs.out_filename = FA_outfile
    tensor2FA.run()    
    
    import nipype.interfaces.mrtrix as mrt
    tensor2vector = mrt.Tensor2Vector()
    tensor2vector.inputs.in_file = infile
    tensor2vector.inputs.out_filename = EV_outfile
    tensor2vector.run()   

    from subprocess import call
    for i in range(1,4):
      command = "tensor_metric " + infile + " -num " + str(i) + " -value " + EV_outfile[:-4] + '_L' + str(i) + '.nii'
      call(command,shell=True)
    return command

def apply_brain_mask(infile,mask_file,outfile):
    import nipype.interfaces.mrtrix as mrt
    MRmult = mrt.MRMultiply()
    MRmult.inputs.in_files = [infile, mask_file]
    MRmult.inputs.out_filename = outfile
    MRmult.run() 

def calculate_other_tensor_metrics(L1_file,L2_file,L3_file,MD_outfile,AD_outfile,RD_outfile):
    # Calculating other tensor metrics
    import nibabel as nib
    import numpy as np
    
    L1_image = nib.load(L1_file)
    L1_data = L1_image.get_data()
    L1_affine = L1_image.get_affine()
    
    L2_image = nib.load(L2_file)
    L2_data = L2_image.get_data()
    L2_affine = L2_image.get_affine()
    
    L3_image = nib.load(L3_file)
    L3_data = L3_image.get_data()
    L3_affine = L3_image.get_affine()
    
    
    # Mean diffusivity
    MD = (L1_data + L2_data + L3_data)/3
    nib.save(nib.Nifti1Image(MD,L1_affine), MD_outfile)
    
    # Axial diffusivity
    AD = L1_data
    nib.save(nib.Nifti1Image(AD,L1_affine), AD_outfile)
    
    # Radial diffusivity
    RD = (L2_data + L3_data)/2
    nib.save(nib.Nifti1Image(RD,L1_affine), RD_outfile)

def preprocess_dwi(subject):
    raw_folder = '/imaging/jb07/CALM/raw_data/DTI/'
    denoised_folder = '/imaging/jb07/CALM/DWI/denoised/'
    eddycorr_folder = '/imaging/jb07/CALM/DWI/eddy_corrected/'
    upsample_folder = '/imaging/jb07/CALM/DWI/upsampled/'
    b0_folder = '/imaging/jb07/CALM/DWI/b0/'
    MNI_coregistered_folder = '/imaging/jb07/CALM/DWI/MNI_coregistered/'
    brain_mask_folder = '/imaging/jb07/CALM/DWI/brain_mask/'
    masked_folder = '/imaging/jb07/CALM/DWI/masked_dwi/'
    dti_folder = '/imaging/jb07/CALM/DWI/dti/'
    FA_folder = '/imaging/jb07/CALM/DWI/FA/'
    EV_folder = '/imaging/jb07/CALM/DWI/EV/'
    masked_FA_folder = '/imaging/jb07/CALM/DWI/FA_masked/'
    masked_EV_folder = '/imaging/jb07/CALM/DWI/EV_masked/'
    MD_folder = '/imaging/jb07/CALM/DWI/MD_masked/'
    AD_folder = '/imaging/jb07/CALM/DWI/RD_masked/'
    RD_folder = '/imaging/jb07/CALM/DWI/AD_masked/'

    import re
    print subject

    # 1. Non-local means denoising
    print 'Denoising'
    denoise_dwi(raw_folder + subject,denoised_folder + subject)
    
    # 2. Eddy current and motion correction
    print 'Motion correction'
    eddy_correction(denoised_folder + subject, eddycorr_folder + subject)

    # 3. Upsampling image
    print 'Upsampling'
    upsample_dwi(eddycorr_folder + subject,upsample_folder + subject)
    
    # 4. Extract b0 image
    print 'Extracting b0'
    extract_b0(upsample_folder + subject,b0_folder + subject)

    # 5. Co-registration with MNI template
    print 'Co-registering with MNI'
    MNI_coregister(b0_folder + subject,upsample_folder + subject,MNI_coregistered_folder + subject)

    # 6. Create brain mask
    print 'Creating brain mask'
    create_brain_mask(b0_folder + subject,brain_mask_folder + subject)
    
    # 7. Fitting the diffusion tensor model
    print 'Fitting DTI model'
    unzip_image_file(upsample_folder + subject)
    fit_diffusion_tensor(upsample_folder + subject,raw_folder + subject[:-7] + '_encoding.txt',dti_folder + subject[:-3])

    # 8. Calculating FA and EV maps
    print 'Calculating metric maps'
    calculate_tensor_metric(dti_folder + subject[:-3],FA_folder + subject[:-3],EV_folder + subject[:-3])
    unzip_image_file(brain_mask_folder + subject[:-7] + '_brain_mask.nii.gz')

    apply_brain_mask(FA_folder + subject[:-3], brain_mask_folder + subject[:-7] + '_brain_mask.nii',masked_FA_folder + subject[:-3])
    apply_brain_mask(EV_folder + subject[:-3], brain_mask_folder + subject[:-7] + '_brain_mask.nii',masked_EV_folder + subject[:-3])
    apply_brain_mask(EV_folder + subject[:-7] + '_L1.nii', brain_mask_folder + subject[:-7] + '_brain_mask.nii',masked_EV_folder + subject[:-7] + '_L1.nii')
    apply_brain_mask(EV_folder + subject[:-7] + '_L2.nii', brain_mask_folder + subject[:-7] + '_brain_mask.nii',masked_EV_folder + subject[:-7] + '_L2.nii')
    apply_brain_mask(EV_folder + subject[:-7] + '_L3.nii', brain_mask_folder + subject[:-7] + '_brain_mask.nii',masked_EV_folder + subject[:-7] + '_L3.nii')

    calculate_other_tensor_metrics(masked_EV_folder + subject[:-7] + '_L1.nii',masked_EV_folder + subject[:-7] + '_L2.nii',masked_EV_folder + subject[:-7] + '_L3.nii',
                                   MD_folder + subject[:-3],RD_folder + subject[:-3],AD_folder + subject[:-3])