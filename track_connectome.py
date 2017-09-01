def get_ROIs(folder,participant):
    # Getting a list of all ROIs
    import os
    import re
    
    track_folder = folder + participant +'/'
    all_files = os.listdir(track_folder)
    ROIs = list()
            
    for single_file in all_files:
        if re.search('.nii',single_file):
            ROIs.append(single_file)
            
    return ROIs

def get_tracks_from_wholebrain_tck(folder,participant,tck_file):
    # Creating files for all tracts
    from IPython.display import clear_output
    import os
    
    ROIs = get_ROIs(folder,participant)
    ROI_folder = folder + participant + '/'
    output_folder = folder + participant + '_tracks/'
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
            output_track = output_folder + roi1[:-13] + '_to_' + roi2[:-13] + '.tck'
            !filter_tracks {whole_brain_tractography} -include {ROI_folder + roi1} -include {ROI_folder + roi2} {output_track}
            clear_output()
            counter += 1

def track_connectome(participant):
    folder = '/imaging/jb07/new_connectome/'
    tck_file = '/imaging/jb07/Denoised_DWI/MRTrix/' + participant + '_whole_brain.tck'

    get_tracks_from_wholebrain_tck(folder,participant,tck_file)