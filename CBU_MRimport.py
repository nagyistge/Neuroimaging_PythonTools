#! /usr/bin/env python
import sys
import optparse

def main():
    p = optparse.OptionParser()
    p.add_option('--outfolder', '-o')
    options, arguments = p.parse_args()
    outfolder = options.outfolder

    def import_data(outfolder):
        #===============================================================
        # Import function
        #===============================================================
        # This function will import data in dicom format from the CBU data repository and convert the files to NifTi format using dcm2nii.
        # The organisation and naming of the data follows the 'Brain Imaging Data Structure (BIDS)' convention. For further information, see bids.neuroimaging.io
        #
        # written by Joe Bathelt, PhD
        # MRC Cognition & Brain Sciences Unit
        # joe.bathelt@mrc-cbu.cam.ac.uk

        import os, re, shutil
        from subprocess import call
        
        folder = '/mridata/cbu/'
        all_files = os.listdir(folder)
        
        for single_file in all_files:
            if re.search('_CALM_STRUCTURALS',single_file) or re.search('_CALM_STRUCTURAL',single_file):
                participant = single_file.split('_')[0]
                top_folder = os.listdir(folder + single_file)
                dicom_folders = os.listdir(folder + single_file + '/' + top_folder[0])
                for dicom_folder in dicom_folders: 
                    
                    #==========================================================================
                    # T1-weighted image
                    if re.search('MPRAGE',dicom_folder):
                        # Creating a folder for the participant's anatomical files if it doesn't exist
                        if not os.path.isdir(outfolder + participant):
                            os.mkdir(outfolder + participant)
                        if not os.path.isdir(outfolder + participant + '/anat/'):
                            os.mkdir(outfolder + participant + '/anat/')
                        
                        # Creating a NifTi version of the T1 volume if it doesn't exist
                        if not os.path.isfile(outfolder + participant + '/anat/' + participant + '_T1w.nii.gz'):
                            participant_folder = folder + single_file + '/' + top_folder[0] + '/' + dicom_folder + '/'
                            files = os.listdir(folder + single_file + '/' + top_folder[0] + '/' + dicom_folder)
                            command = 'dcm2nii -g y -a y -o ' + outfolder + participant + '/anat/ ' + folder + single_file + '/' + top_folder[0] + '/' + dicom_folder +'/' + files[0]
                            call(command,shell=True)
                            
                            # Renaming the file according to BIDS convention
                            for afile in os.listdir(outfolder + participant + '/anat/'):
                                if re.search('co',outfolder + participant + '/anat/' + afile):
                                    os.rename(outfolder + participant + '/anat/' + afile,outfolder + participant + '/anat/' + participant + '_T1w.nii.gz')
                                if re.search('MPRAGE',outfolder + participant + '/anat/' + afile) and not re.search('co',outfolder + participant + '/anat/' + afile):
                                    os.remove(outfolder + participant + '/anat/' + afile)
                       
                    #==========================================================================
                    # T2-weighted image
                    
                    if re.search('TSE',dicom_folder):
                        # Creating a folder for the participant's anatomical files if it doesn't exist
                        if not os.path.isdir(outfolder + participant):
                            os.mkdir(outfolder + participant)
                        if not os.path.isdir(outfolder + participant + '/anat/'):
                            os.mkdir(outfolder + participant + '/anat/')
                        
                        # Creating a NifTi version of the T2 volume if it doesn't exist
                        if not os.path.isfile(outfolder + participant + '/anat/' + participant + '_T2map.nii.gz'):
                            participant_folder = folder + single_file + '/' + top_folder[0] + '/' + dicom_folder + '/'
                            files = os.listdir(folder + single_file + '/' + top_folder[0] + '/' + dicom_folder)
                            command = 'dcm2nii -g y -a y -o ' + outfolder + participant + '/anat/ ' + folder + single_file + '/' + top_folder[0] + '/' + dicom_folder +'/' + files[0]
                            call(command,shell=True)
                            
                            # Renaming the file according to BIDS convention
                            for afile in os.listdir(outfolder + participant + '/anat/'):
                                if re.search('TSE',afile):
                                    os.rename(outfolder + participant + '/anat/' + afile,outfolder + participant + '/anat/' + participant + '_T2map.nii.gz')
        
                       
                    #==========================================================================
                    # Diffusion-weighted image
        
                    if re.search('DTI',dicom_folder) and not re.search('FA',dicom_folder) and not re.search('TRACEW',dicom_folder):
                        # Creating a folder for the participant's anatomical files if it doesn't exist
                        if not os.path.isdir(outfolder + participant):
                            os.mkdir(outfolder + participant)
                        if not os.path.isdir(outfolder + participant + '/dwi/'):
                            os.mkdir(outfolder + participant + '/dwi/')
                        
                        # Creating a NifTi version of the T1 volume if it doesn't exist
                        if not os.path.isfile(outfolder + participant + '/dwi/' + participant + '_dwi.nii.gz',):
                            shutil.copytree(folder + single_file + '/' + top_folder[0] + '/' + dicom_folder +'/',outfolder + participant + '/dwi/temp/')
                            participant_folder = folder + single_file + '/' + top_folder[0] + '/' + dicom_folder + '/'
                            files = os.listdir(folder + single_file + '/' + top_folder[0] + '/' + dicom_folder)
                            command = 'dcm2nii -g y -a y -o ' + outfolder + participant + '/dwi/ ' + outfolder + participant + '/dwi/temp/' + files[0]
                            print command 
        
                            call(command,shell=True)
                            
                            # Renaming the file according to BIDS convention
                            for afile in os.listdir(outfolder + participant + '/dwi/'):
                                if re.search('.nii.gz',outfolder + participant + '/dwi/' + afile):
                                    os.rename(outfolder + participant + '/dwi/' + afile,outfolder + participant + '/dwi/' + participant + '_dwi.nii.gz')
                                if re.search('.bval',outfolder + participant + '/dwi/' + afile):
                                    os.rename(outfolder + participant + '/dwi/' + afile,outfolder + participant + '/dwi/' + participant + '_dwi.bval')
                                if re.search('.bvec',outfolder + participant + '/dwi/' + afile):
                                    os.rename(outfolder + participant + '/dwi/' + afile,outfolder + participant + '/dwi/' + participant + '_dwi.bvec')
                            
                            shutil.rmtree(outfolder + participant + '/dwi/temp/')
        
                    #==========================================================================
                    # Resting-state
                    if re.search('EPI',dicom_folder):
                        # Creating a folder for the participant's anatomical files if it doesn't exist
                        if not os.path.isdir(outfolder + participant):
                            os.mkdir(outfolder + participant)
                        if not os.path.isdir(outfolder + participant + '/func/'):
                            os.mkdir(outfolder + participant + '/func/')
                        
                        # Creating a NifTi version of the T1 volume if it doesn't exist
                        if not os.path.isfile(outfolder + participant + '/func/' + participant + '_task-rest_bold.nii.gz'):
                            participant_folder = folder + single_file + '/' + top_folder[0] + '/' + dicom_folder + '/'
                            files = os.listdir(folder + single_file + '/' + top_folder[0] + '/' + dicom_folder)
                            command = 'dcm2nii -g y -a y -o ' + outfolder + participant + '/func/ ' + folder + single_file + '/' + top_folder[0] + '/' + dicom_folder +'/' + files[0]
                            call(command,shell=True)
                            
                            # Renaming the file according to BIDS convention
                            for afile in os.listdir(outfolder + participant + '/func/'):
                                os.rename(outfolder + participant + '/func/' + afile,outfolder + participant + '/func/' + participant + '_task-rest.nii.gz')

    try:
        import_data(outfolder)
    except:
        print 'ERROR'
        
if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())   