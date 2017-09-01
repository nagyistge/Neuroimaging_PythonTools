#! /usr/bin/env python
import sys
import optparse

def main():
    p = optparse.OptionParser()
    p.add_option('--outfolder', '-o')
    options, arguments = p.parse_args()
    outfolder = options.outfolder

    def get_behavioural_measure(variable_name, behavioural_data_file):
        """ Obtain a dataframe of behavioural data
        This function takes a column name and csv file with behavioural data and returns a dataframe with data from that variable.

        Parameters
        ------------
        variable_name: string,
            column name that refers to the behavioural variable of interest
        behavioural_data_file: string,
            filename of the csv file with the behavioural data

        Returns
        -------
        behavioural_df : pandas dataframe
            dataframe with the data of the behavioural variable in one column
        """
        import sys
        import pandas as pd

        data = pd.read_csv(behavioural_data_file)

        # Seleting only the relevant column
        try:
            data = data[['ID No.',variable_name]]
            data.columns = ['ID','measure']
            return data
        except:
            print('Variable name not found\n')
            print('These variables are available:\n')
            print(list(data.columns.values))

    def get_behavioural_measure_for_ID(variable_name, behavioural_data_file, ID):
        """ Obtain behavioural data for a specific participant
        This function takes a variable name and the filename of a csv file with behavioural data and
        returns the behavioural data for the participant with the specied ID.

        Parameters
        ------------
        variable_name: string,
            column name that refers to the behavioural variable of interest
        behavioural_data_file: string,
            filename of the csv file with the behavioural data
        ID: string,
            ID of the participant

        Returns
        -------
        behavioural_df : pandas dataframe
            dataframe with the data of the behavioural variable in one column.
            NB: If the participant cannot be found in the behavioural data, the functio return 999
        """

        behavioural_df = get_behavioural_measure(variable_name,behavioural_data_file)
        behavioural_df = behavioural_df[behavioural_df['ID'] == ID]
        try:
            return behavioural_df['measure'].values[0]
        except:
            return 999

    def get_behavioural_ID(ID_string):
        """ Get behavioural ID for an imaging ID
        This function takes a CALM imaging ID and matches it to the corresponding behavioural ID

        Parameters
        ------------
        ID_string: string,
        imaging ID, e.g. 'CBU16001'

        Returns
        -------
        behavioural ID : string
        CALM behavioural ID corresponding to the imaging ID
        NB: If the participant cannot be found in the behavioural data, the functio return 999
        """

        # Reading data from the lookup table
        import pandas as pd
        data = pd.Series.from_csv('/imaging/jb07/CALM/MRI ID match Oct16.csv')

        # Removing brackts
        import re
        for counter in range(1,len(data)):
            entry = data[counter]
            if re.search(r'\[(){}[]]+',str(entry)):
                data[counter] = entry[1:-1]

        # Getting the matching ID
        try:
            ID = data[data == ID_string].index.tolist()
            return int(ID[0])
        except:
            print 'Not Found: ' + ID_string
            return 999

    def get_behavioural_measure_for_imaging_ID(variable_name, behavioural_data_file, MRI_ID):
        """ Obtain behavioural data for a specific participant using the imaging ID
        This function takes a variable name and the filename of a csv file with behavioural data and
        returns the behavioural data for the participant with the specied ID.

        Parameters
        ------------
        variable_name: string,
            column name that refers to the behavioural variable of interest
        behavioural_data_file: string,
            filename of the csv file with the behavioural data
        ID: string,
            MRI ID of the participant, e.g 'CBU160001'

        Returns
        -------
        value : float
            dataframe with the data of the behavioural variable in one column.
            NB: If the participant cannot be found in the behavioural data, the functio return 999
        """
        import pandas as pd

        try:
            value = get_behavioural_measure_for_ID(variable_name, behavioural_data_file, get_behavioural_ID(MRI_ID))
            return value
        except:
            print variable_name

    def get_demographic_data(behavioural_data_file, MRI_ID):
        """ This function collects demographic information (date of birth, date of test, age, gender) in a dictionary

        Parameters
        ------------
        behavioural_data_file: string,
            filename of the csv file with the behavioural data
        MRI_ID: string,
            MRI ID of the participant, e.g 'CBU160001'

        Returns
        -------
        demographics : dictionary
            dictionary with demographic information
        """

        demographics = dict()
        demographics['CALM_ID'] = get_behavioural_ID(MRI_ID)
        demographics['MRI_ID'] = MRI_ID
        demographics['D.O.B'] = get_behavioural_measure_for_imaging_ID('D.O.B', behavioural_data_file, MRI_ID)
        demographics['Date of test'] = get_behavioural_measure_for_imaging_ID('D.O.B', behavioural_data_file, MRI_ID)
        demographics['Age_in_months'] = get_behavioural_measure_for_imaging_ID('Age_in_months', behavioural_data_file, MRI_ID)
        demographics['Gender'] = get_behavioural_measure_for_imaging_ID('Gender', behavioural_data_file, MRI_ID)
        return demographics

    def get_behavioural_data(behavioural_data_file, MRI_ID):
        """ This function collects results from behavioural testing based on the CALM neuropsych battery

        Parameters
        ------------
        behavioural_data_file: string,
            filename of the csv file with the behavioural data
        MRI_ID: string,
            MRI ID of the participant, e.g. 'CBU160001'

        Returns
        -------
        behavioural_data : dictionary
            dictionary with results of the behavioural testing
        """

        behavioural_data = dict()
        measures = ['Assessed?', 'ID No.', 'Unnamed: 2', 'Date of test', 'D.O.B', 'AGE', 'Age_in_months', 'Gender(1=1)', 'Referrer_Code', 'ADHD Code', 'Diagnosis', 'ADD', 'ADHD', 'Dyslexia', 'Dyspraxia', 'Dysgraphia', 'Dyscalculia', 'FASD', 'Generalised_dev_delay', 'Global_delay', 'Social_anxiety_disorder', 'Depression', 'Autism', 'Tourettes', 'DAMP', 'Anxiety', 'OCD', 'Hyperactivity', 'ADHD Medicated', 'Possible ADHD (under assesment)', 'Under SLT', 'Primary_Reason', 'Detailed_Reason', 'Matrix_Reasoning_Raw ', 'Matrix_Reasoning_T', 'Matrix_Reasoning_Percentile', 'Matrix_Reasoning_Raw_Closestage', 'Matrix_Reasoning_T_Closestage', 'Matrix_Reasoning_per_Closestage', 'Matrix_Reasoning_T_Score_for_analysis', 'PPVT_Raw', 'PPVT_Std', 'PPVT_Percentile', 'WIAT_Spelling_Raw', 'WIAT_Spelling_std', 'WIAT_Spelling_Percentile', 'WIAT_Reading_Raw', 'WIAT_Reading_Std', 'WIAT_Reading_Percentile', 'WIAT_Numerical_raw', 'WIAT_Numerical_Std', 'WIAT_Numerical_Percentile', 'WJ_Math_fluency_raw', 'WJ_Math_Std', 'WJ_Math_fluency_Percentile', 'Maths_standard_score_for_analysis', 'PhAB_Alliteration_Raw', 'PhAB_Alliteration_Std', 'PhAB_Alliteration_Percentile', 'PhAB_Alliteration_raw_closestage', 'PhAB_Alliteration_std_closestage', 'PhAB_Alliteration_percentile_closestage', 'PhAB_Picture_Alliteration_raw', 'PhAB_Picture_Alliteration_Std', 'PhAB_Picture_Alliteration_Percentile', 'PhAB_Picture_Alliteration_raw_closestage', 'PhAB_Picture_Alliteration_std_closestage', 'PhAB_Picture_Alliteration_percentile_closestage', 'PhAb_Allieration_Standard_Score_For_Analysis', 'PhAB_Object_RAN_RT_raw', 'PhAB_Object_RAN_RT_std', 'PhAB_Object_RAN_RT_percentile', 'PhAB_Picture_Alliteration_raw_closestage.1', 'PhAB_Picture_Alliteration_std_closestage.1', 'PhAB_Picture_Alliteration_percentile_closestage.1', 'PhAB_Object_RAN_RT_Standard_Score_for_analysis', 'AWMA_Digit_Recall_Raw', 'AWMA_Digit_Recall_Standard', 'AWMA_Digit_Recall_Percentile', 'AWMA_Dot_Matrix_Raw', 'AWMA_Dot_Matrix_Standard', 'AWMA_Dot_Matrix_Percentile', 'AWMA_Backward_Digit__Raw', 'AWMA_Backward_Digit__Standard', 'AWMA_Backward_Digit__Percentile', 'AWMA_Mr_X__Raw', 'AWMA_Mr_X__Standard', 'AWMA_Mr_X__Percentile', 'AWMA_Mr_X_processing', 'CMS_immediate_raw', 'CMS_immediate_Scaled', 'CMS_immediate_percentile_rank', 'CMS_delayed_raw', 'CMS_delayed_Scaled', 'CMS_delayed_percentile_rank', 'CMS_delayed_recognition__raw', 'CMS_delayed_recognition__Scaled', 'CMS_delayed_recognition__percentile_rank', 'CMS_immediate_thematic_raw', 'CMS_immediate_thematic_Scaled', 'CMS_immediate_thematic_percentile_rank', 'CMS_delayed_thematic_raw', 'CMS_delayed_thematic_Scaled', 'CMS_delayed_thematic_percentile_rank', 'Teach2_cancellation_raw', 'Teach2_cancellation_percentile', 'Teach2_SART_raw', 'Teach2_SART_percentile', 'Teach2_SART_Coefficient of variation_percent', 'Teach2_vigilance_raw', 'Teach2_vigilance_percentile', 'Teach2_RBBS_Switching_ %increase_RT ', 'Teach2_RBBS_Switching_ %increase_RT_percentile', 'Teach2_RBBS_Switching_Rtcost', 'Teach2_RBBS_Switching_Rtcost_percentile', 'Teach2_int', 'Tower_total_raw', 'Tower_total_scaled', '%Tower_total', 'Tower_meanfirstmove_ratio_raw', 'Tower_meanfirstmove_scaled', 'Tower_ruleviolations', 'Tower_ruleviolations_scaled', 'Number_Letter_Switching_Raw', 'Number_Letter_Switching_Scaled', 'Number_Letter_Switching_%', 'Difference_switching_and_number+letterseq', 'Difference_switching_and_number+letterseq_Scaledscore', 'Difference_switching_and_number+letterseq_%', 'FollowingInstructions_Features', 'FollowingInstructions_Items', 'FollowingInstructions_Actions', 'FollowingInstructions_TrialsCorrect', 'FollowingInstructions_Span', 'CNRep_total', 'CNRep_2', 'CNRep_3', 'CNRep_4', 'CNRep_5', 'SDQ_total', 'int_SDQ_total', 'SDQ_emotion', 'intSDQ_emotion', 'SDQ_conduct', 'int_SDQ_Conduct', 'SDQ_Hyperactivity', 'int_SDQ_Hyperactivity', 'SDQ_Peerproblems', 'int_SDQ_Peerproblems', 'SDQ_prosocial', 'int_SDQ_prosocial', 'Conners_inattention_raw', 'Conners_inattention_T', 'Conners_inattention_percentile', 'Conners_hyperactivity_impulsivity_raw', 'Conners_hyperactivity_impulsivity_T', 'Conners_hyperactivity_impulsivity_percentile', 'Conners_learning_problems_raw', 'Conners_learning_problems_T', 'Conners_learning_problems_percentile', 'Conners_ExecutiveFunction_raw', 'Conners_ExecutiveFunction_T', 'Conners_ExecutiveFunction_percentile', 'Conners_agression_raw', 'Conners_agression_T', 'Conners_agression_percentile', 'Conners_PeerRelations_raw', 'Conners_PeerRelations_T', 'Conners_PeerRelations_percentile', 'Conners_Positive_Impression', 'Conners_Positive_Impression_interpretation', 'Connors_Negative_Impression', 'Connors_Negative_Impression_interpretation', 'Brief_Inhibit_raw', 'Brief_Inhibit_T', 'Inhibit_Brief_percentile', 'Brief_Shift_raw', 'Brief_Shift_T', 'Shift_Brief_percentile', 'Brief_Emotional_Control_raw', 'Brief_Emotional_Control_T', 'Emotional_Control_Brief_percentile', 'Brief_Initiate_raw', 'Brief_Initiate_T', 'Initiate_Brief_percentile', 'Brief_Working_Memory_raw', 'Brief_Working_Memory_T', 'Working_Memory_Brief_percentile', 'Brief_Planning_raw', 'Brief_Planning_T', 'Planning_Brief_percentile', 'Brief_Organisation_raw', 'Brief_Organisation_T', 'Organisation_Brief_percentile', 'Brief_Monitor_raw', 'Brief_Monitor_T', 'Monitor_Brief_percentile', 'BRIEF_Behavior_Regulation_Index_Raw', 'BRIEF_Behavior_Regulation_Index_T', 'BRIEF_Behavior_Regulation_Index_percentile', 'BRIEF_Metacognition_Index_raw', 'BRIEF_Metacognition_Index_T', 'BRIEF_Metacognition_Index_[ercentile', 'BRIEF_Global_Executive_Composite_Raw', 'BRIEF_Global_Executive_Composite_T', 'BRIEF_Global_Executive_Composite_percentile', 'CCC_speech_raw', 'CCC_speech_std', 'CCC_speech_percentile', 'CCC_syntax_raw', 'CCC_syntax_std', 'CCC_syntax_percentile', 'CCC_semantic_raw', 'CCC_semantic_std', 'CCC_semantic_percentile', 'CCC_coherence_raw', 'CCC_coherence_std', 'CCC_coherence_percentile', 'CCC_ippropriate_initiation_raw', 'CCC_ippropriate_initiation_std', 'CCC_ippropriate_initiation_percentile', 'CCC_stereo_raw', 'CCC_stereo_std', 'CCC_stereo_percentile', 'CCC_context_raw', 'CCC_context_std', 'CCC_context_percentile', 'CCC_nonverbal_raw', 'CCC_nonverbal_std', 'CCC_nonverbal_percentile', 'CCC_social_raw', 'CCC_social_std', 'CCC_social_percentile', 'CCC_interests_raw', 'CCC_interests_std', 'CCC_interests_percentile', 'CCC_Global_raw', 'CCC_Global_percentile', 'CCC_Valid']

        for measure in measures:
            behavioural_data[measure] = get_behavioural_measure_for_imaging_ID(measure, behavioural_data_file, MRI_ID)

        return behavioural_data

    def behavioural_data_for_BIDS(base_directory, behavioural_data_file, MRI_ID):
        """ This function creates text files with demographic information and results of behavioural testing for a BIDS data structure

        Parameters
        ------------
        base_directory: string,
            full path to the BIDS folder
        behavioural_data_file: string,
            filename of the csv file with the behavioural data
        MRI_ID: string,
            MRI ID of the participant, e.g 'CBU160001'

        Returns
        -------
        behavioural_data : dictionary
            dictionary with results of the behavioural testing
        """

        import os
        import json

        # Writing demographic data
        if not os.path.isfile(base_directory + MRI_ID + '/' + MRI_ID + '_demographics.txt'):
            demographics = get_demographic_data(behavioural_data_file, MRI_ID)
            import json
            with open(base_directory + MRI_ID + '/' + MRI_ID + '_demographics.txt', 'w+') as outfile:
                json.dump(demographics, outfile)

        # Writing behavioural data
        if not os.path.isdir(base_directory + MRI_ID + '/beh/'):
            os.mkdir(base_directory + MRI_ID + '/beh/')

        behavioural_data = get_behavioural_data(behavioural_data_file, MRI_ID)
        import json
        with open(base_directory + MRI_ID + '/beh/' + MRI_ID + '_beh.txt', 'w+') as outfile:
            json.dump(behavioural_data, outfile)

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
        all_files = reversed(sorted(os.listdir(folder)))

        for single_file in all_files:
            if re.search('_CALM_STRUCTURALS',single_file) or re.search('_CALM_STRUCTURAL',single_file):
                participant = single_file.split('_')[0]
                if not os.path.isdir(outfolder + participant): os.mkdir(outfolder + participant)
                #behavioural_data_for_BIDS(outfolder, '/imaging/jb07/CALM/CALM_data_May2016.csv', participant)
                top_folder = os.listdir(folder + single_file)
                dicom_folders = os.listdir(folder + single_file + '/' + top_folder[0])

                if not os.path.isfile(outfolder + participant + '/anat/' + participant + '_T1w.nii.gz'):
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

    import_data(outfolder)

if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())
