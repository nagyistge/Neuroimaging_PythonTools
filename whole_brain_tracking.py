import nipype.interfaces.mrtrix as mrt
import os
folders = ['c1']#,'c2','c3','c5','c6','c7','c8','z1','z2','z3','z4']

for participant in folders:
    # Generating broad white matter mask
    os.remove('/home/jb07/Documents/Denoised_DWI/MRTrix/' + participant + '_wm_mask.nii')
    threshold_wmmask = mrt.Threshold()
    threshold_wmmask.inputs.absolute_threshold_value = 0.4
    threshold_wmmask.inputs.in_file = '/home/jb07/Documents/Denoised_DWI/MRTrix/' + participant + '_FA.nii'
    threshold_wmmask.inputs.out_filename = '/home/jb07/Documents/Denoised_DWI/MRTrix/' + participant + '_wm_mask.nii'
    threshold_wmmask.run()
    
    ## Performing whole-brain tracking
    sdprobtrack = mrt.ProbabilisticSphericallyDeconvolutedStreamlineTrack()
    sdprobtrack.inputs.in_file = '/home/jb07/Documents/Denoised_DWI/MRTrix/' + participant + '_CSD8.nii'
    sdprobtrack.inputs.seed_file = '/home/jb07/Documents/Denoised_DWI/MRTrix/' + participant + '_wm_mask.nii'
    sdprobtrack.inputs.mask_file = '/home/jb07/Documents/Denoised_DWI/MRTrix/' + participant + '_mask.nii'
    sdprobtrack.inputs.out_file = '/home/jb07/Documents/Denoised_DWI/MRTrix/' + participant + '_whole_brain.tck'
    sdprobtrack.inputs.inputmodel = 'SD_PROB'
    sdprobtrack.inputs.desired_number_of_tracks = 150000
    sdprobtrack.run()

    ## Converting to TrackVis format
    import nipype.interfaces.mrtrix as mrt
    tck2trk = mrt.MRTrix2TrackVis()
    tck2trk.inputs.in_file = '/home/jb07/Documents/Denoised_DWI/MRTrix/' + participant + '_whole_brain.tck'
    tck2trk.inputs.image_file = '/home/jb07/Documents/Denoised_DWI/MRTrix/' + participant + '_FA.nii'
    tck2trk.inputs.out_filename = '/home/jb07/Documents/Denoised_DWI/MRTrix/' + participant + '_whole_brain.trk'
    tck2trk.run()

    #streamtrack SD_PROB '/home/jb07/Documents/Denoised_DWI/MRTrix/c1_CSD8.nii' --seed '/home/jb07/Documents/Denoised_DWI/MRTrix/c1_wm_mask.nii' --mask '/home/jb07/Documents/Denoised_DWI/MRTrix/c1_mask.nii' -num 150000 '/home/jb07/Documents/Denoised_DWI/MRTrix/c1_whole_brain.tck'