def run_FSL_anat(subject_list,base_directory,out_directory):
    #==============================================================
    # Loading required packages
    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe
    import nipype.interfaces.utility as util
    from nipype.interfaces import fsl
    import nipype.interfaces.dipy as dipy
    from nipype import SelectFiles
    import os

    # Getting the subject ID
    infosource  = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),name='infosource')
    infosource.iterables = ('subject_id', subject_list)

    # Getting the relevant diffusion-weighted data
    templates = dict(T1='{subject_id}/func/{subject_id}_task-rest.nii.gz')

    selectfiles = pe.Node(SelectFiles(templates),
                       name="selectfiles")
    selectfiles.inputs.base_directory = os.path.abspath(base_directory)

    # Slice time correction
    slicetime_correction = pe.Node(interface=fsl.SliceTimer(time_repetition=2,interleaved=True),name='slicetime_correction')

    # Extract the middle volume for re-alignment
    extract_ref = pe.Node(interface=fsl.ExtractROI(t_size=1,t_min=135),name = 'extractref')

    # Motion correction through co-registration to middle volume
    mcflirt = pe.Node(interface=fsl.MCFLIRT(save_mats = True,save_plots = True),name='mcflirt')

    # Interpolating to 2mm isotropic resolution
    resample = pe.Node(interface=dipy.Resample(interp=3,vox_size=(2.,2.,2.)), name='resample')

    # Calculating a mean image 
    mean_image = pe.Node(interface=fsl.ImageMaths(op_string = '-Tmean', suffix='_mean'),name='mean_image')

    # Creating a brain mask based on the mean image
    bet = pe.Node(interface=fsl.BET(mask = True, no_output=True, frac = 0.3),name='bet')

    # Mask the functional image
    apply_mask = pe.Node(interface=fsl.ImageMaths(op_string='-mas'),name='apply_mask')

    # Co-register functional to structural image
    flt = pe.Node(interface=fsl.FLIRT(dof=6, cost_func='corratio'),name='flirt')
    flt.inputs.reference = '/imaging/jb07/CALM/EPI/SPM_EPI_template.nii'

    # Applying the transform
    applyxfm = pe.Node(interface=fsl.ApplyXfm(apply_xfm = True),name='applyxfm')
    applyxfm.inputs.reference = '/imaging/jb07/CALM/EPI/SPM_EPI_template.nii'
    
    # Setting up the workflow
    fMRI_preproc = pe.Workflow(name='fMRI_PreProc')

    # Reading in files
    fMRI_preproc.connect(infosource, 'subject_id', selectfiles, 'subject_id')
    fMRI_preproc.connect(selectfiles, 'T1', slicetime_correction, 'in_file')

    fMRI_preproc.connect(slicetime_correction, 'slice_time_corrected_file', extract_ref, 'in_file')
    fMRI_preproc.connect(slicetime_correction, 'slice_time_corrected_file', mcflirt, 'in_file')
    fMRI_preproc.connect(extract_ref, 'roi_file', mcflirt, 'ref_file')

    fMRI_preproc.connect(mcflirt, 'out_file', resample, 'in_file')
    fMRI_preproc.connect(resample, 'out_file', mean_image, 'in_file')
    fMRI_preproc.connect(mean_image, 'out_file', bet, 'in_file')
    fMRI_preproc.connect(resample, 'out_file',apply_mask, 'in_file')
    fMRI_preproc.connect(bet, 'mask_file', apply_mask, 'in_file2')

    fMRI_preproc.connect(apply_mask, 'out_file', flt, 'in_file')
    fMRI_preproc.connect(mcflirt, 'out_file', applyxfm, 'in_file')
    fMRI_preproc.connect(flt, 'out_matrix_file', applyxfm, 'in_matrix_file')

    # Running the workflow
    fMRI_preproc.base_dir = os.path.abspath(out_directory)
    fMRI_preproc.write_graph()
    fMRI_preproc.run()