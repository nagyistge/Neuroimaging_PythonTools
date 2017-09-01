def run_eddy(acqparams, bvecs, bvals, dwi, index_file, output_folder, subject_id):

    from nipype.interfaces import fsl
    import nipype.pipeline.engine as pe
    import os

    output_folder = output_folder + subject_id + '/'

    os.mkdir(output_folder)
    os.chdir(output_folder)


    fslroi = fsl.ExtractROI()
    fslroi.inputs.in_file = dwi
    fslroi.inputs.t_min = 0
    fslroi.inputs.t_size = 1
    fslroi.inputs.roi_file = output_folder + '/b0.nii.gz'
    fslroi.run()

    # Create a brain mask
    bet = fsl.BET()
    bet.inputs.in_file = output_folder + '/b0.nii.gz'
    bet.inputs.frac=0.3
    bet.inputs.robust=False
    bet.inputs.mask=True
    bet.inputs.no_output=False
    bet.inputs.out_file = output_folder + '/b0_brain.nii.gz'
    bet.run()

    # Eddy-current and motion correction
    eddy = pe.Node(interface=fsl.epi.Eddy(args='-v'), name='eddy')
    eddy.inputs.in_acqp  = acqparams
    eddy.inputs.in_bvec  = bvecs
    eddy.inputs.in_bval  = bvals
    eddy.inputs.in_file = dwi
    eddy.inputs.in_index = index_file
    eddy.inputs.in_mask = output_folder + '/b0_brain_mask.nii.gz'
    eddy.inputs.out_base = output_folder + subject_id
    eddy.run()

    return output_folder + subject_id + '.eddy_movement_rms'

def get_displacement(eddy_movement_file):
    import pandas as pd

    df = pd.read_csv(eddy_movement_file, delim_whitespace=True, names=['absolute','relative'])

    max_displacement = df.max()['relative']

    return max_displacement

def get_motion_parameters_BIDS(acqparams, index_file, input_folder, output_folder):

    import nibabel as nib
    import os
    import pandas as pd
    import re

    df = pd.DataFrame()
    subject_list = sorted([subject for subject in os.listdir(input_folder) if re.search('CBU', subject)])

    for subject in subject_list[1:]:

        try:
            number_of_volumes = nib.load(input_folder + subject + '/dwi/' + subject + '_dwi.nii.gz').shape[3]
        except:
            number_of_volumes = 999

        if not os.path.isfile(output_folder + subject) and number_of_volumes == 69:
            dwi = input_folder + subject + '/dwi/' + subject + '_dwi.nii.gz'
            bvecs = input_folder + subject + '/dwi/' + subject + '_dwi.bvec'
            bvals = input_folder + subject + '/dwi/' + subject + '_dwi.bval'

            run_eddy(acqparams, bvecs, bvals, dwi, index_file, output_folder, subject)

        if os.path.isfile(output_folder + subject + '/subject.eddy_movement_rms'):
            try:
                max_displacement = get_displacement(output_folder + subject + '/subject.eddy_movement_rms')
            except:
                max_displacement = 999

        df.set_value(subject, 'movement', max_displacement)
        df.set_value(subject, 'volumes', number_of_volumes)
        df.to_csv(output_folder + '/Displacement_Results.csv')
