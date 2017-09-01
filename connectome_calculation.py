def get_ROIs(folder,participant):
    # Getting a list of all ROIs
    import os
    import re
    
    track_folder = folder + participant
    all_files = os.listdir(folder)
    ROIs = list()
    
    for single_file in all_files:
        if re.search('.nii',single_file):
            ROIs.append(single_file)
            
    return ROIs

def get_tracks_from_wholebrain_tck(ROI_folder,participant,tck_file):
    # Creating files for all tracts
    from IPython.display import clear_output
    import os
    
    ROIs = get_ROIs(ROI_folder,participant)
    output_folder = '/imaging/jb07/new_connectome/' + participant + '_tracks/'
    if os.path.isdir(output_folder) == False:
        os.mkdir(output_folder)
    total_number = len(ROIs)*len(ROIs)
    counter = 1;
    
    for ROI in ROIs:
        roi1 = ROI
        
        for ROI in ROIs:
            print str(counter) + '/' + str(total_number)
            roi2 = ROI
            whole_brain_tractography = tck_file
            output_track = output_folder + roi1[:-4] + '_to_' + roi2[:-4] + '.tck'
            import nipype.interfaces.mrtrix as mrt
            filt = mrt.FilterTracks()
            filt.inputs.in_file = whole_brain_tractography
            filt.inputs.include_file = ROI_folder + roi1
            filt.inputs.args = '-include ' + ROI_folder + roi2
            filt.inputs.out_file = output_track
            filt.run()    
            clear_output()
            counter += 1    

def connectome_calculation(participant):
    folder = '/imaging/jb07/new_connectome/'
    tck_file = '/imaging/jb07/Denoised_DWI/MRTrix/' + participant + '_whole_brain.tck'
    get_tracks_from_wholebrain_tck('/imaging/jb07/resting_state/ROIs/' + participant + '_vols/',participant,tck_file)