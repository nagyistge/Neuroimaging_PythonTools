def trk_MNI_coregistration(trackfile,FA_file,out_FA_file,out_FA_matrix,out_trackfile):
    # Co-registring the participant's data to the MNI template
    from nipype.interfaces import fsl
    flt = fsl.FLIRT(dof=12, cost_func='corratio')
    flt.inputs.in_file = FA_file
    flt.inputs.reference = '/imaging/jb07/ZDHHC9/FreeSurfer/fsaverage/mri/brain.nii.gz'
    flt.inputs.out_file = out_FA_file
    flt.inputs.out_matrix_file = out_FA_matrix
    flt.run()
    
    # Moving the whole-brain streamlines from participant space to MNI space using the transformation matrix from step 1
    from subprocess import call 
    command = 'track_transform ' + trackfile + ' -src ' + FA_file + ' -ref ' + reference + ' -reg ' + flt.inputs.out_matrix_file + " -reg_type 'flirt' " + out_trackfile
    call(command,shell=True)