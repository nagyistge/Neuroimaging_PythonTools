#! /usr/bin/env python
import optparse
import os
import re
import shutil
from subprocess import call
import sys
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

    folder = '/mridata/cbu'
    all_files = reversed(sorted(os.listdir(folder)))
    outfolder = '/imaging/jb07/CALM/CALM_BIDS'
    behavioural_data_file = '/imaging/jb07/CALM/CALM_data_May2016.csv'
    ID_match_file = '/imaging/jb07/CALM/MRI ID match Oct16.csv'

    for single_file in all_files:
        if re.search('_CALM_', single_file):
            participant = single_file.split('_')[0]
            print(participant)
            create_folder(join([outfolder, participant]))
            write_BIDS_behaviour(participant, outfolder,
                                 behavioural_data_file, ID_match_file)

            folders = join([folder, single_file, os.listdir(
                join([folder, single_file]))[0]])

            modalities = ['MPRAGE', 'TSE', 'EPI', 'DTI']
            BIDS_subfolders = {'MPRAGE': 'anat/',
                               'TSE': 'anat/', 'DTI': 'dwi/', 'EPI': 'func/'}
            BIDS_extensions = {'MPRAGE': '_T1w', 'TSE': '_T2map',
                               'DTI': '_dwi', 'EPI': '_task-rest'}

            for subject_folder in os.listdir(folders):
                for modality in modalities:
                    if re.search(modality, subject_folder):
                        if modality == 'DTI' and re.search('Series10', subject_folder):
                            export_folder = join(
                                [outfolder, participant, BIDS_subfolders[modality]])
                            participant_folder = join(
                                [folders, subject_folder])
                            try:
                                import_dwi(participant,
                                    participant_folder, export_folder, BIDS_extensions[modality])
                            except:
                                print('FAILED IMPORT')
                        else:
                            export_folder = join(
                                [outfolder, participant, BIDS_subfolders[modality]])
                            participant_folder = join(
                                [folders, subject_folder])
                            try:
                                import_data(participant,
                                    participant_folder, export_folder, BIDS_extensions[modality])
                            except:
                                print('FAILED IMPORT')

    print('Done!')

if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())
