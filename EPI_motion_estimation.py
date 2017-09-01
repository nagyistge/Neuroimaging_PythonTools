def get_displacements(input_image, subject, out_folder):
    # ===============================================================#
    # Caculating the RMS displacement time course
    # for method see:http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLMotionOutliers 
    # and http://www.brainmapping.org/NITP/images/Summer2013Slides/Rissman_artifact_detection_NITP_2013.pdf
    # ===============================================================#
    """
    inputs:
    reference_image: image that was used for coregistration
    parameter_folder: transformation matrices that for moving each image to the reference image
    
    outputs:
    displacement_overview: displacement between consecutive volumes
    max_displacement: maximal displacement in mm
    mean_displacement: mean displacement in mm
    """
    
    import os 
    import numpy as np
    from subprocess import call
    
    if not os.path.isfile(out_folder + subject + '_metrics.txt'):
        command = 'fsl_motion_outliers -i ' + input_image + ' ' + \
                    '-o ' + out_folder + subject + '_confound_matrix ' + \
                    '-s ' + out_folder + subject + '_metrics.txt ' + \
                    '-p ' + out_folder + subject + '_metrics.png ' + \
                    '--fd'
        call(command, shell=True)

    return np.sqrt(np.mean(np.loadtxt(out_folder + subject + '_metrics.txt')**2))

def get_motion_parameters_BIDS(in_folder,out_folder):
    # ===============================================================#
    # Calculate displacements in a number of DWI files 
    # ===============================================================#
    """
    inputs:
    in_folder: folder containing the DWI volumes
    out_folder: folder where the output should be directed

    outputs:
    Displacement_Results.csv - csv file with the maximum displacement for each participant, which will be saved in the output directory
    histogramme showing the distribution of maximum displacements
    """

    import sys
    import os
    import re
    import numpy as np
    import nibabel as nib
    import pandas as pd

    os.chdir(out_folder)
    files = os.listdir(in_folder)
    rms_displacements = list()
    names = list()
    time_points = list()

    subjects = sorted(os.listdir(in_folder))

    for subject in subjects: 
            if os.path.isfile(in_folder + subject + '/func/' + subject + '_task-rest.nii.gz'):
                rms_displacements.append(get_displacements(in_folder + subject + '/func/' + subject + '_task-rest.nii.gz', subject, out_folder))
                time_points.append(nib.load(in_folder + subject + '/func/' + subject + '_task-rest.nii.gz').header.get_data_shape()[3])
                names.append(subject)

    # Writing results to a spreadsheet
    dataframe = pd.DataFrame(np.vstack([rms_displacements, time_points]).T,index=names,columns=['rms displacement [mm]','number of volumes'])
    dataframe = dataframe.sort()
    dataframe.to_csv(out_folder + 'Displacements_Results.csv')
    
    print('Number of datasets with more than 0.5mm displacement: ' + str(len(dataframe[dataframe['rms displacement [mm]'] > 0.5])))

