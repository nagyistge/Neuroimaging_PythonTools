#! /usr/bin/env python
import sys
sys.path.append('/home/jb07/nipype_installation/')

import optparse
import os
import re
import shutil
from subprocess import call
from nipype.interfaces.dcm2nii import Dcm2nii

def create_folder(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

join = lambda input_list: '/'.join(input_list)

def import_data(participant, participant_folder, export_folder, extension):

    if not os.path.isfile(join([export_folder, participant + extension + '.nii.gz'])):
        create_folder(export_folder)

        converter = Dcm2nii()
        converter.inputs.source_dir = participant_folder
        converter.inputs.gzip_output = True
        converter.inputs.output_dir = export_folder
        converter.inputs.source_in_filename = True
        converter.run()

        for out_file in os.listdir(export_folder):
            if not re.search(r'^CBU', out_file):
                os.rename(join([export_folder, out_file]), join(
                    [export_folder, participant + extension + '.nii.gz']))

def import_dwi(participant, participant_folder, export_folder, extension):
    import os
    from shutil import copytree, rmtree

    if not os.path.isfile(join([export_folder, participant + extension + '.nii.gz'])):

        create_folder(export_folder)

        copytree(participant_folder, export_folder + '/temp')

        converter = Dcm2nii()
        converter.inputs.source_dir = export_folder + '/temp'
        converter.inputs.gzip_output = True
        converter.inputs.output_dir = export_folder
        converter.inputs.source_in_filename = True
        converter.run()

        rmtree(export_folder + 'temp/')

        for out_file in os.listdir(export_folder):
            if not re.search(r'^CBU', out_file):
                if re.search(r'.nii.gz$', out_file):
                    os.rename(join([export_folder, out_file]), join(
                        [export_folder, participant + extension + '.nii.gz']))
                if re.search(r'.bval$', out_file):
                    os.rename(join([export_folder, out_file]), join(
                        [export_folder, participant + extension + '.bval']))
                if re.search(r'.bvec$', out_file):
                    os.rename(join([export_folder, out_file]), join(
                        [export_folder, participant + extension + '.bvec']))

def get_behavioural_ID(ID_string, ID_match_file):
    import pandas as pd
    data = pd.Series.from_csv(ID_match_file)

    # Removing brackts
    import re
    for counter in range(1, len(data)):
        entry = data[counter]
        if re.search(r'\[(){}[]]+', str(entry)):
            data[counter] = entry[1:-1]

    # Getting the matching ID
    try:
        ID = data[data == ID_string].index.tolist()
        return int(ID[0])
    except:
        print 'Not Found: ' + ID_string
        return 999

def write_BIDS_behaviour(MRI_ID, base_directory, behavioural_data_file, ID_match_file):
    import json
    import sys
    import pandas as pd

    data = pd.read_csv(behavioural_data_file)
    data = data[data['ID No.'] ==
                get_behavioural_ID(MRI_ID, ID_match_file)]
    data['Gender'] = data['Gender(1=1)']

    if not data.empty:
        if not os.path.isdir(base_directory + '/' + MRI_ID + '/beh/'):
            os.mkdir(base_directory + '/' + MRI_ID + '/beh/')

        with open(base_directory + '/' + MRI_ID + '/' + MRI_ID + '_demographics.txt', 'w+') as outfile:
            json.dump(data[['ID No.', 'D.O.B', 'Date of test', 'Age_in_months', 'Gender']].to_dict(
                'record')[0], outfile)

        with open(base_directory + '/' + MRI_ID + '/beh/' + MRI_ID + '_beh.txt', 'w+') as outfile:
            json.dump(data.to_dict('record')[0], outfile)

def main():
    p = optparse.OptionParser()
    p.add_option('--outfolder', '-o')
    options, arguments = p.parse_args()
    outfolder = options.outfolder

    import os
    import re

    in_folder = '/mridata/cbu/'
    CALM_files = sorted([folder for folder in os.listdir(in_folder) if re.search('_CALM_', folder)])

    for CALM_file in CALM_files[::-1]:
        subject = CALM_file.split('_')[0]
        filename_base = in_folder + CALM_file + '/' + os.listdir(in_folder + CALM_file)[0] + '/'
        DTI_folder = filename_base + 'Series013_CBU_DTI_64InLea_2x2x2_32chn'
        EPI_folder = filename_base + 'Series022_CBU_EPI_Standard_32chn_B270'
        T1_folder = filename_base + 'Series010_CBU_MPRAGE_32chn'
        T2_folder = filename_base + 'Series011_CBU_TSE_32chn'

        subfolders = os.listdir(filename_base)
        for subfolder in subfolders:
            if (re.search('DTI', subfolder)) and not ((re.search('TRACEW', subfolder)) or (re.search('FA', subfolder))):
                DTI_folder = filename_base + subfolder
            if re.search('EPI', subfolder):
                EPI_folder = filename_base + subfolder
            if re.search('MPRAGE', subfolder):
                T1_folder = filename_base + subfolder
            if re.search('TSE', subfolder):
                T2_folder = filename_base + subfolder

        if not os.path.isdir(outfolder + subject):
            os.mkdir(outfolder + subject)

        # DTI
        export_folder = outfolder + subject + '/dwi/'
        subject_folder = DTI_folder
        if os.path.isdir(DTI_folder) and not os.path.isfile(export_folder + subject + '_dwi.nii.gz'):
            try:
                import_dwi(subject, subject_folder, export_folder, '_dwi')
            except:
                print('failed import ' + subject)

        # EPI
        export_folder = outfolder + subject + '/func/'
        subject_folder = EPI_folder
        if os.path.isdir(EPI_folder) and not os.path.isfile(export_folder + subject + '_task-rest.nii.gz'):
            try:
                import_data(subject, subject_folder, export_folder, '_task-rest')
            except:
                print('failed import ' + subject)

        # T1
        export_folder = outfolder + subject + '/anat/'
        subject_folder = T1_folder
        if os.path.isdir(T1_folder) and not os.path.isfile(export_folder + subject + '_T1w.nii.gz'):
            try:
                import_data(subject, subject_folder, export_folder, '_T1w')
            except:
                print('failed import ' + subject)

        # T2
        export_folder = outfolder + subject + '/anat/'
        subject_folder = T2_folder
        if os.path.isdir(T2_folder) and not os.path.isfile(export_folder + subject + '_T2map.nii.gz'):
            try:
                import_data(subject, subject_folder, export_folder, '_T2map')
            except:
                print('failed import ' + subject)

if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())
