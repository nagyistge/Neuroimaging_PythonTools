import sys
sys.path.append('/home/jb07/nipype_installation/')

import nibabel as nib
import os
import re
from subprocess import call

# ======================================================================
# Computing displacements using FSL eddy
# ======================================================================
"""
This part creates scripts to calculate the between-volume displacements with FSL's eddy. These script are submitted to the cluster for fast parallel processing
"""

in_folder = '/imaging/jb07/CALM/CALM_BIDS/'
out_folder = '/imaging/jb07/CALM/DWI/motion_estimation/scripts/'
subject_list = sorted([subject for subject in os.listdir(in_folder) if re.search('CBU', subject)])

for subject in subject_list[4:]:
        try:
            number_of_volumes = nib.load(in_folder + subject + '/dwi/' + subject + '_dwi.nii.gz').shape[3]
        except:
            number_of_volumes = 999

        if number_of_volumes == 69:
            # Writing the python script
            cmd = "import sys \n" + \
            "sys.path.append('/home/jb07/joe_python/') \n" + \
            "sys.path.append('/home/jb07/nipype_installation/') \n" + \
            "import DWI_motion_estimation_V2 \n" + \
            "subject = '" + subject + "'\n" + \
            "acqparams = '/imaging/jb07/CALM/CALM_BIDS/acqparams.txt' \n" + \
            "index_file = '/imaging/jb07/CALM/CALM_BIDS/index.txt' \n" + \
            "bvals = '/imaging/jb07/CALM/CALM_BIDS/" + subject + "/dwi/" + subject + "_dwi.bval' \n" + \
            "bvecs = '/imaging/jb07/CALM/CALM_BIDS/" + subject + "/dwi/" + subject + "_dwi.bvec'\n" + \
            "dwi = '/imaging/jb07/CALM/CALM_BIDS/" + subject + "/dwi/" + subject + "_dwi.nii.gz'\n" + \
            "output_folder = '/imaging/jb07/CALM/DWI/motion_estimation/' \n" + \
            "DWI_motion_estimation_V2.run_eddy(acqparams, bvecs, bvals, dwi, index_file, output_folder, subject)"

            file = open(out_folder + subject + '_motion_estimation.py', 'w')
            file.write(cmd)
            file.close()

            # Writing a wrapper bash script
            cmd = "python " + out_folder + subject + '_motion_estimation.py'
            file = open(out_folder + subject + '_motion_estimation.sh', 'w')
            file.write(cmd)
            file.close()

            # Submitting the shell script to the compute cluster
            cmd = "qsub " + out_folder + subject + '_motion_estimation.sh'
            call(cmd, shell=True)

# ======================================================================
# Generate overview CSV file
# ======================================================================
"""
This part of the script collects the displacement results and creates a csv file
"""
import sys
sys.path.append('/home/jb07/nipype_installation/')

import nibabel as nib
import os
import pandas as pd

folder = '/imaging/jb07/CALM/DWI/motion_estimation/'
filename = lambda subject: folder + subject + '/' + subject + '.eddy_movement_rms'
image_filename = lambda subject: folder + subject + '/' + subject + '.nii.gz'

subject_list = sorted([subject for subject in os.listdir(folder) if os.path.isfile(filename(subject))])
df = pd.DataFrame(index=subject_list)

for subject in subject_list:
    subject_df = pd.read_csv(filename(subject), delim_whitespace=True, names=['absolute','relative'])
    max_displacement = subject_df.max()['relative']
    number_of_volumes = nib.load(image_filename(subject)).shape[3]

    df.set_value(subject, 'movement', max_displacement)
    df.set_value(subject, 'volumes', number_of_volumes)

df.to_csv(folder + '/Displacement_Results.csv')
