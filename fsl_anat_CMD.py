#! /usr/bin/env python
import sys
import optparse

def main():
    from subprocess import call
    import os 

    p = optparse.OptionParser()
    p.add_option('--infolder', '-i')
    p.add_option('--outfolder', '-o')
    options, arguments = p.parse_args()
    output_folder = options.outfolder
    input_folder = options.infolder

    subjects = os.listdir(input_folder)

    for subject in subjects:
        print subject

        if os.path.isfile(input_folder + subject + '/anat/' + subject + '_T1w.nii.gz') and not os.path.isdir(output_folder + subject + '.anat'):
            command = 'fsl_anat --noreg --nononlinreg --noseg --nosubcortseg -i ' + input_folder + subject + '/anat/' + subject + '_T1w.nii.gz -o ' + output_folder + subject
            call(command,shell=True)

if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())   