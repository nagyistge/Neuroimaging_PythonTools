def call_mcflirt(input_image,outfolder):
    # ===============================================================#
    # Use McFlirt to calculate the rigid body alignment between volumes in a 4D image file
    # for method see: http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MCFLIRT
    # ===============================================================#
    """
    inputs:
    input_image: 4D input image for alignment

    outputs:
    McFlirt generates outputfolders for realigned image and motion parameters
    """

    from subprocess import call
    command = 'mcflirt -in ' + input_image + ' -plots -stats -mats -o ' + outfolder
    call(command,shell=True)


def get_displacements(reference_image,parameter_folder):
    # ===============================================================#
    # Caculating the RMS displacement time course
    # for method see: http://www.fmrib.ox.ac.uk/analysis/techrep/tr99mj1/tr99mj1/index.html
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
    from subprocess import check_output

    parameter_files = os.listdir(parameter_folder)
    parameter_files = np.sort(parameter_files)
    number_of_files = len(parameter_files)
    displacements = list()

    for i in range(0,number_of_files-1):
        command = 'rmsdiff ' + parameter_folder + parameter_files[i] + ' ' + parameter_folder + parameter_files[i+1] + ' ' + reference_image
        displacement = check_output(command,shell=True)
        displacement = float(displacement[:-2])
        displacements.append(displacement)

    return displacements, np.max(displacements), np.mean(displacements)

def plot_displacements(displacements):
    # ===============================================================#
    # Plot mm displacements vs volume number
    # ===============================================================#
    """
    inputs:
    displacements: array with displacement in mm between consecutive volumes
    """

    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(displacements)
    plt.axhline(np.mean(displacements),color='k',ls='dashed')
    plt.plot(np.argmax(displacements),np.max(displacements),'or')
    plt.ylabel('RMS displacement in mm')
    plt.xlabel('Volume')
    return plt

def plot_displacement_histogramme(max_displacements):
    # ===============================================================#
    # Plot a histogramme of the maximum displacements
    # ===============================================================#
    """
    inputs:
    max_displacements: list of maximum displacement values
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # the histogram of the data
    n, bins, patches = plt.hist([round(elem, 0) for elem in max_displacements ], round(max(max_displacements)), facecolor='green', alpha=0.75)
    plt.xlabel('maximum displacement [mm]')
    plt.ylabel('number of cases')
    plt.grid(True)
    plt.show()

def get_motion_parameters(in_folder,out_folder):
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
    import pandas as pd

    os.chdir(out_folder)
    files = os.listdir(in_folder)
    mean_displacements = list()
    max_displacements = list()
    names = list()

    for afile in files:
        if re.search('.nii.gz',afile):
            afile = afile.split('.')
            afile = afile[0]
            names.append(afile)

            call_mcflirt(in_folder + afile,out_folder + afile)
            displacements,max_displacement,mean_displacement = get_displacements(out_folder + afile + '_meanvol.nii.gz',out_folder + afile + '.mat/')
            mean_displacements.append(mean_displacement)
            max_displacements.append(max_displacement)


    # Writing results to a spreadsheet
    dataframe = pd.DataFrame(max_displacements,index=names,columns=['maximum displacement [mm]'])
    dataframe = dataframe.sort()
    dataframe.to_csv(out_folder + 'Displacements_Results.csv')

    print('Number of datasets with more than 4mm displacement: ' + str(len(dataframe[dataframe['maximum displacement [mm]'] > 4])))

    # Printing a histogramme of the maximum displacement distribution
    import numpy as np
    import matplotlib.pyplot as plt

    # the histogram of the data
    plot_displacement_histogramme(max_displacements)

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

    import nibabel as nib
    import os
    import re
    import sys
    import pandas as pd

    os.chdir(out_folder)
    files = os.listdir(in_folder)
    mean_displacements = list()
    max_displacements = list()
    numbers_of_volumes = list()
    names = list()

    subjects = sorted(os.listdir(in_folder))

    for subject in subjects:
        print subject
        if os.path.isfile(in_folder + subject + '/dwi/' + subject + '_dwi.nii.gz'):
            names.append(subject)
            try:
                number_of_volumes = nib.load(in_folder + subject + '/dwi/' + subject + '_dwi.nii.gz').shape[3]

                if not os.path.isdir('/imaging/jb07/CALM/DWI/motion_estimation/' + subject + '.mat'):
                    if number_of_volumes == 69:
                        call_mcflirt(in_folder + subject + '/dwi/' + subject + '_dwi.nii.gz',out_folder + subject)
                    else:
                        mean_displacements.append(999)
                        max_displacements.append(999)

                displacements,max_displacement,mean_displacement = get_displacements(out_folder + subject + '_meanvol.nii.gz',out_folder + subject + '.mat/')
                mean_displacements.append(mean_displacement)
                max_displacements.append(max_displacement)
                numbers_of_volumes.append(number_of_volumes)
            except:
                mean_displacements.append(999)
                max_displacements.append(999)
                numbers_of_volumes.append(999)

    # Writing results to a spreadsheet
    dataframe = pd.DataFrame(index=names)
    dataframe['displacement'] = max_displacements
    dataframe['volumes'] = numbers_of_volumes
    dataframe = dataframe.sort()
    dataframe.to_csv(out_folder + 'Displacements_Results.csv')

    # Printing a histogramme of the maximum displacement distribution
    import numpy as np
    import matplotlib.pyplot as plt

    # the histogram of the data
    plot_displacement_histogramme(max_displacements)

