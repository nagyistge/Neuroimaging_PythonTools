def get_behavioural_measure(variable_name):
    import pandas as pd
    import sys
    sys.path.append('/home/jb07/python_modules/lib/python/')
    # Reading SPSS sav file to a pandas dataframe
    #from savReaderWriter import SavReader
    #raw_data = SavReader("/imaging/jb07/CALM/Reduced_CALM_Cognitive_data_cleaned_NO_OUTLIERS_JH_FW_15Sept2015_C_factor_scores_subgroups_gender_15thOctober_2015.sav", returnHeader=True)
    #raw_data_list = list(raw_data) # this is slow
    #data = pd.DataFrame(raw_data_list) # this is slow
    #data = pd.read_csv('/imaging/jb07/CALM/CALM_data_Dec2015.csv')
    data = pd.read_csv('/imaging/jb07/CALM/CALM_data_May2016.csv')
    #data = data.rename(columns=data.loc[0]).iloc[1:] # setting columnheaders, this is slow too.

    # Seleting only the relevant column
    try:
        data = data[['ID No.',variable_name]]
        data.columns = ['ID','measure']
        return data
    except:
        print('Variable name not found\n')
        print('These variables are available:\n')
        print(list(data.columns.values))

def get_PBS_queue(status):
    import pandas as pd
    from subprocess import check_output

    try:
        output = check_output('qselect -u jb7 | xargs qstat', shell=True).split('\n')
        output = [entry.split() for entry in output[2:-1]]
        df = pd.DataFrame()

        counter = 0;
        for entry in output:
            df.set_value(counter, 'ID', entry[2])
            df.set_value(counter, 'status', entry[4])
            counter += 1

        df = df[df['ID'] == 'jb07']
        df = df[df['status'] == status]
        return len(df)

    except:
        return 0

def plot_histogram(data):
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    n, bins, patches = plt.hist(data, 50, facecolor='blue', alpha=0.75)
    plt.xlabel('Score')
    plt.ylabel('Cases')
    plt.axis()
    plt.grid(True)
    plt.show()

def mean_scale_data(data):
    import numpy as np

    scaled_data = data
    mean = np.mean(data)

    for i in range(0,len(data)):
        scaled_data[i] = data[i] - mean

    return scaled_data

def write_randomise_matrix(data,filename):
    import numpy as np
    file = open(filename, "w")
    dimensions = data.shape

    file.write('/NumWaves ' + str(dimensions[1]) + '\n')
    file.write('/NumPoints ' + str(len(data)) + '\n')
    ppheight = str()
    for j in range(0,dimensions[1]):
        ppheight += str(np.max(data[:,j])) + ' '
    file.write('/PPheights ' + ppheight + '\n')
    file.write('/Matrix\n')

    for i in range(0,dimensions[0]):
        line = str()
        for j in range(0,dimensions[1]):
            line += str(data[i,j]) + ' '
        file.write(line + '\n')


def write_randomise_con(data,filename):
    import numpy as np
    file = open(filename, "w")
    dimensions = data.shape

    file.write('/ContrastName1 positivecorr' + '\n')
    file.write('/ContrastName2 negativecorr' + '\n')
    file.write('/NumWaves ' + str(dimensions[1]) + '\n')
    file.write('/NumContrasts 2' + '\n')
    ppheight = str()
    for j in range(0,dimensions[1]):
        ppheight += str(np.max(data[:,j])) + ' '
    file.write('/PPheights ' + ppheight + '\n')
    file.write('/Matrix\n')
    file.write('1 0\n')
    file.write('-1 0\n')

def get_behavioural_IDs_for_FA_files(folder):
    import os
    import re

    files = os.listdir(folder)

    MRI_IDs = list()
    for afile in files:
        if re.search('CBU',afile):
            ID = afile.split('.')
            MRI_IDs.append(ID[0])

    behavioural_IDs = list()
    for MRI_ID in MRI_IDs:
        behavioural_IDs.append(get_behavioural_ID(MRI_ID))

    if sum([i if i==999 else 0 for i in behavioural_IDs]) > 0:
        del behavioural_IDs[behavioural_IDs.index(999)]
    return behavioural_IDs

def get_behavioural_IDs_for_imaging_IDs(MRI_IDs):
    import os
    import re

    behavioural_IDs = list()
    for MRI_ID in MRI_IDs:
        behavioural_IDs.append(get_behavioural_ID(MRI_ID))

    if sum([i if i==999 else 0 for i in behavioural_IDs]) > 0:
        del behavioural_IDs[behavioural_IDs.index(999)]
    return behavioural_IDs

def get_behavioural_measure_for_ID(variable_name,ID):
    behavioural_df = get_behavioural_measure(variable_name)
    behavioural_df = behavioural_df[behavioural_df['ID'] == ID]
    try:
        return behavioural_df['measure'].values[0]
    except:
        return 999


def get_behavioural_measure_for_imaging_IDs(variable_name,MRI_IDs):
    import pandas as pd
    measures = list()

    for MRI_ID in MRI_IDs:
        measures.append(get_behavioural_measure_for_ID(variable_name,get_behavioural_ID(MRI_ID)))

    df = pd.DataFrame({'MRI.ID':MRI_IDs,variable_name:measures})
    return df


def get_behavioural_measure_for_FA_files(variable_name,folder):
    import pandas as pd
    behavioural_measure = get_behavioural_measure(variable_name)
    behavioural_IDs = get_behavioural_IDs_for_FA_files(folder)
    mask = behavioural_measure['ID'].isin(behavioural_IDs)
    behavioural_measure = behavioural_measure[mask]
    behavioural_measure = behavioural_measure.set_index('ID')

    behavioural_IDs = behavioural_measure.index.values
    imaging_IDs = list()

    for behavioural_ID in behavioural_IDs:
        imaging_IDs.append(get_imaging_ID(str(int(behavioural_ID))))

    behavioural_measure['MRI.ID'] = imaging_IDs
    return behavioural_measure


def get_behavioural_data_for_IDs(variable_name,behavioural_IDs):
    import pandas as pd
    behavioural_measure = get_behavioural_measure(variable_name)
    mask = behavioural_measure['ID'].isin(behavioural_IDs)
    behavioural_measure = behavioural_measure[mask]
    behavioural_measure = behavioural_measure.set_index('ID')

    behavioural_IDs = behavioural_measure.index.values
    imaging_IDs = list()

    for behavioural_ID in behavioural_IDs:
        imaging_IDs.append(get_imaging_ID(str(int(behavioural_ID))))

    behavioural_measure['MRI.ID'] = imaging_IDs
    return behavioural_measure

def get_df_of_behavioural_data(measure):
    import pandas as pd
    df = get_behavioural_measure(measure)
    behavioural_IDs = df['ID']
    imaging_IDs = list()

    for behavioural_ID in behavioural_IDs:
       imaging_IDs.append(get_imaging_ID(str(int(behavioural_ID))))

    df['MRI.ID'] = imaging_IDs
    df = df[pd.isnull(df['MRI.ID']) == False]
    df = df[df['MRI.ID'] != 'notFound']

    return df

def copy_BIDS_files(source_directory,source_final_step,file_identifier,target_folder):
    """
    source_directory: the root directory that contains the subject folders
    source_final_step: name of the folder that contains the last processing step for each subject
    file_identifier: extension that identifies the folder in the final processing step folder, e.g. '_FA.nii.gz'
    target_folder: folder that will contain the copied files
    """
    import os, re, shutil

    for folder in os.listdir(source_directory):
        if os.path.isdir(source_directory + '/' + folder):
            if os.path.isdir(source_directory + '/' + folder + '/' + source_final_step):
                for afile in os.listdir(source_directory + '/' + folder + '/' + source_final_step):
                    if re.search(file_identifier,afile):
                        print source_directory + '/' + folder + '/' + source_final_step + '/' + afile
                        shutil.copyfile(source_directory + '/' + folder + '/' + source_final_step + '/' + afile,target_folder + afile)


def get_list_of_BIDS_IDs(source_directory,file_tag):
    import os, re

    if file_tag == 'T1w':
        files = list()
        for subfolder in os.listdir(source_directory):
            if re.search('CBU',subfolder):
                if os.path.isfile(source_directory + subfolder + '/anat/' + subfolder + '_T1w.nii.gz'):
                    files.append(source_directory + subfolder + '/anat/' + subfolder + '_T1w.nii.gz')

    if file_tag == 'dwi':
        files = list()
        for subfolder in os.listdir(source_directory):
            if re.search('CBU',subfolder):
                if os.path.isfile(source_directory + subfolder + '/dwi/' + subfolder + '_dwi.nii.gz'):
                    files.append(source_directory + subfolder + '/dwi/' + subfolder + '_dwi.nii.gz')

    if file_tag == 'task-rest':
        files = list()
        for subfolder in os.listdir(source_directory):
            if re.search('CBU',subfolder):
                if os.path.isfile(source_directory + subfolder + '/func/' + subfolder + '_task-rest.nii.gz'):
                    files.append(source_directory + subfolder + '/func/' + subfolder + '_task-rest.nii.gz')

    subject_list = [subject.split('/')[-1].split('_')[0] for subject in files]
    return sorted(subject_list)

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
    measures = ['Date of test','D.O.B','AGE','Age_in_months','Gender(1=1)','Referrer_Code','ADHD Code','Diagnosis','ADD','ADHD','Dyslexia','Dyspraxia','Dysgraphia','Dyscalculia','FASD','Generalised_dev_delay','Global_delay','Social_anxiety_disorder','Depression','Autism','Tourettes','DAMP','Anxiety','OCD','Hyperactivity','ADHD Medicated','Possible ADHD (under assesment)','Under SLT','Primary_Reason','Detailed_Reason','Matrix_Reasoning_Raw ','Matrix_Reasoning_T','Matrix_Reasoning_Percentile','Matrix_Reasoning_Raw_Closestage','Matrix_Reasoning_T_Closestage','Matrix_Reasoning_per_Closestage','Matrix_Reasoning_T_Score_for_analysis','PPVT_Raw','PPVT_Std','PPVT_Percentile','WIAT_Spelling_Raw','WIAT_Spelling_std','WIAT_Spelling_Percentile','WIAT_Reading_Raw','WIAT_Reading_Std','WIAT_Reading_Percentile','WIAT_Numerical_raw','WIAT_Numerical_Std','WIAT_Numerical_Percentile','WJ_Math_fluency_raw','WJ_Math_Std','WJ_Math_fluency_Percentile','Maths_standard_score_for_analysis','PhAB_Alliteration_Raw','PhAB_Alliteration_Std','PhAB_Alliteration_Percentile','PhAB_Alliteration_raw_closestage','PhAB_Alliteration_std_closestage','PhAB_Alliteration_percentile_closestage','PhAB_Picture_Alliteration_raw','PhAB_Picture_Alliteration_Std','PhAB_Picture_Alliteration_Percentile','PhAB_Picture_Alliteration_raw_closestage','PhAB_Picture_Alliteration_std_closestage','PhAB_Picture_Alliteration_percentile_closestage','PhAb_Allieration_Standard_Score_For_Analysis','PhAB_Object_RAN_RT_raw','PhAB_Object_RAN_RT_std','PhAB_Object_RAN_RT_percentile','PhAB_Picture_Alliteration_raw_closestage','PhAB_Picture_Alliteration_std_closestage','PhAB_Picture_Alliteration_percentile_closestage','PhAB_Object_RAN_RT_Standard_Score_for_analysis','AWMA_Digit_Recall_Raw','AWMA_Digit_Recall_Standard','AWMA_Digit_Recall_Percentile','AWMA_Dot_Matrix_Raw','AWMA_Dot_Matrix_Standard','AWMA_Dot_Matrix_Percentile','AWMA_Backward_Digit__Raw','AWMA_Backward_Digit__Standard','AWMA_Backward_Digit__Percentile','AWMA_Mr_X__Raw','AWMA_Mr_X__Standard','AWMA_Mr_X__Percentile','AWMA_Mr_X_processing','CMS_immediate_raw','CMS_immediate_Scaled','CMS_immediate_percentile_rank','CMS_delayed_raw','CMS_delayed_Scaled','CMS_delayed_percentile_rank','CMS_delayed_recognition__raw','CMS_delayed_recognition__Scaled','CMS_delayed_recognition__percentile_rank','CMS_immediate_thematic_raw','CMS_immediate_thematic_Scaled','CMS_immediate_thematic_percentile_rank','CMS_delayed_thematic_raw','CMS_delayed_thematic_Scaled','CMS_delayed_thematic_percentile_rank','Teach2_cancellation_raw','Teach2_cancellation_percentile','Teach2_SART_raw','Teach2_SART_percentile','Teach2_SART_Coefficient of variation_percent','Teach2_vigilance_raw','Teach2_vigilance_percentile','Teach2_RBBS_Switching_ %increase_RT ','Teach2_RBBS_Switching_ %increase_RT_percentile','Teach2_RBBS_Switching_Rtcost','Teach2_RBBS_Switching_Rtcost_percentile','Teach2_int','Tower_total_raw','Tower_total_scaled','%Tower_total','Tower_meanfirstmove_ratio_raw','Tower_meanfirstmove_scaled','Tower_ruleviolations','Tower_ruleviolations_scaled','Number_Letter_Switching_Raw','Number_Letter_Switching_Scaled','Number_Letter_Switching_%','Difference_switching_and_number+letterseq','Difference_switching_and_number+letterseq_Scaledscore','Difference_switching_and_number+letterseq_%','FollowingInstructions_Features','FollowingInstructions_Items','FollowingInstructions_Actions','FollowingInstructions_TrialsCorrect','FollowingInstructions_Span','CNRep_total','CNRep_2','CNRep_3','CNRep_4','CNRep_5','SDQ_total','int_SDQ_total','SDQ_emotion','intSDQ_emotion','SDQ_conduct','int_SDQ_Conduct','SDQ_Hyperactivity','int_SDQ_Hyperactivity','SDQ_Peerproblems','int_SDQ_Peerproblems','SDQ_prosocial','int_SDQ_prosocial','Conners_inattention_raw','Conners_inattention_T','Conners_inattention_percentile','Conners_hyperactivity_impulsivity_raw','Conners_hyperactivity_impulsivity_T','Conners_hyperactivity_impulsivity_percentile','Conners_learning_problems_raw','Conners_learning_problems_T','Conners_learning_problems_percentile','Conners_ExecutiveFunction_raw','Conners_ExecutiveFunction_T','Conners_ExecutiveFunction_percentile','Conners_agression_raw','Conners_agression_T','Conners_agression_percentile','Conners_PeerRelations_raw','Conners_PeerRelations_T','Conners_PeerRelations_percentile','Conners_Positive_Impression','Conners_Positive_Impression_interpretation','Connors_Negative_Impression','Connors_Negative_Impression_interpretation','Brief_Inhibit_raw','Brief_Inhibit_T','Inhibit_Brief_percentile','Brief_Shift_raw','Brief_Shift_T','Shift_Brief_percentile','Brief_Emotional_Control_raw','Brief_Emotional_Control_T','Emotional_Control_Brief_percentile','Brief_Initiate_raw','Brief_Initiate_T','Initiate_Brief_percentile','Brief_Working_Memory_raw','Brief_Working_Memory_T','Working_Memory_Brief_percentile','Brief_Planning_raw','Brief_Planning_T','Planning_Brief_percentile','Brief_Organisation_raw','Brief_Organisation_T','Organisation_Brief_percentile','Brief_Monitor_raw','Brief_Monitor_T','Monitor_Brief_percentile','BRIEF_Behavior_Regulation_Index_Raw','BRIEF_Behavior_Regulation_Index_T','BRIEF_Behavior_Regulation_Index_percentile','BRIEF_Metacognition_Index_raw','BRIEF_Metacognition_Index_T','BRIEF_Metacognition_Index_[ercentile','BRIEF_Global_Executive_Composite_Raw','BRIEF_Global_Executive_Composite_T','BRIEF_Global_Executive_Composite_percentile','CCC_speech_raw','CCC_speech_std','CCC_speech_percentile','CCC_syntax_raw','CCC_syntax_std','CCC_syntax_percentile','CCC_semantic_raw','CCC_semantic_std','CCC_semantic_percentile','CCC_coherence_raw','CCC_coherence_std','CCC_coherence_percentile','CCC_ippropriate_initiation_raw','CCC_ippropriate_initiation_std','CCC_ippropriate_initiation_percentile','CCC_stereo_raw','CCC_stereo_std','CCC_stereo_percentile','CCC_context_raw','CCC_context_std','CCC_context_percentile','CCC_nonverbal_raw','CCC_nonverbal_std','CCC_nonverbal_percentile','CCC_social_raw','CCC_social_std','CCC_social_percentile','CCC_interests_raw','CCC_interests_std','CCC_interests_percentile','CCC_Global_raw','CCC_Global_percentile','CCC_Valid']

    for measure in measures:
        behavioural_data[measure] = get_behavioural_measure_for_imaging_ID(measure, behavioural_data_file, MRI_ID)

    return behavioural_data


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

def get_number_of_images(folder):
    import os

    T1_files = 0
    DWI_files = 0
    Func_files = 0

    for participant in os.listdir(folder):
        if os.path.isfile(folder + participant + '/anat/' + participant + '_T1w.nii.gz'):
            T1_files += 1

        if os.path.isfile(folder + participant + '/dwi/' + participant + '_dwi.nii.gz'):
            DWI_files += 1

        if os.path.isfile(folder + participant + '/func/' + participant + '_task-rest.nii.gz'):
            Func_files += 1

    print('T1-weighted images:' + str(T1_files))
    print('Diffusion-weighted images:' + str(DWI_files))
    print('BOLD-weighted images:' + str(Func_files))

def get_cluster_jobs():
    import subprocess

    if subprocess.check_output("qselect -u jb07 -s R", shell=True):
        jobs = subprocess.check_output("qselect -u jb07 -s R | xargs qstat", shell=True).split('\n')[2:-1]
        number_of_running_jobs = len(jobs)
    else:
        number_of_running_jobs = 0

    if subprocess.check_output("qselect -u jb07 -s H", shell=True):
        jobs = subprocess.check_output("qselect -u jb07 -s H | xargs qstat", shell=True).split('\n')[2:-1]
        number_of_held_jobs = len(jobs)
    else:
        number_of_held_jobs = 0

    if subprocess.check_output("qselect -u jb07 -s Q", shell=True):
        jobs = subprocess.check_output("qselect -u jb07 -s Q | xargs qstat", shell=True).split('\n')[2:-1]
        number_of_queued_jobs = len(jobs)
    else:
        number_of_queued_jobs = 0

    print('Number of running jobs: ' + str(number_of_running_jobs))
    print('Number of held jobs: ' + str(number_of_held_jobs))
    print('Number of queued jobs: ' + str(number_of_queued_jobs))

    return (number_of_running_jobs, number_of_held_jobs, number_of_queued_jobs)

def get_behavioural_ID(ID_string):
    # Reading data from the lookup table
    import pandas as pd
    data = pd.Series.from_csv('/imaging/jb07/CALM/MRI ID match new.csv')

    # Removing brackts
    import re
    for counter in range(1,len(data)):
        entry = data[counter]
        if re.search(r"[(){}[\]]+",str(entry)):
            data[counter] = entry[1:-1]

    # Getting the matching ID
    try:
        ID = data[data == ID_string].index.tolist()
        return int(ID[0])
    except:
        print 'Not Found: ' + ID_string
        return 999


def get_imaging_ID(ID_string):
    # Reading data from the lookup table
    import pandas as pd
    data = pd.Series.from_csv('/imaging/jb07/CALM/MRI ID match Aug17.csv')

    # Removing brackts
    import re
    for counter in range(1,len(data)):
        entry = data[counter]
        if re.search(r"[(){}[\]]+",str(entry)):
            data[counter] = entry[1:-1]

    # Getting the matching ID
    try:
        ID = data[data.index == ID_string].values.tolist()
        return ID[0]
    except:
        return float('nan')
