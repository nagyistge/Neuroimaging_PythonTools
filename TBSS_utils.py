#TBSS_utils
def demean_variable(variable):
    import numpy as np
    return variable - np.mean(variable)

def write_randomise_matrix_age_effect(in_file,filename):
    df = pd.read_csv(in_file)
    variables = ['Age_in_months','Movement','Gender']
    df = df[variables]

    for variable in variables[:3]:
        df[variable] = demean_variable(df[variable].values)

    stats_data = np.vstack([df['Age_in_months'].values,
                  df['Movement'].values,
                  df['Gender'].values,
                  np.ones(len(df))]).T

    print df.corr()

    file = open(filename,'w')
    file.write('/NumWaves ' + str(stats_data.shape[1]) + '\n')
    file.write('/NumPoints ' + str(stats_data.shape[0]) + '\n')
    file.write('/PPheights ' + ' '.join(map(str, np.max(stats_data,axis=0))) + '\n')
    file.write('/Matrix \n')
    [file.write(' '.join(map(str,stats_data[i,...])) + '\n') for i in range(0,stats_data.shape[0])];
    file.close()

# Factor score effect
import pandas as pd 
import numpy as np

def write_randomise_matrix_with_age(in_file,in_variable,filename):
    import pandas as pd 
    import numpy as np
    df = pd.read_csv(in_file)
    variables = [in_variable] + ['Age_in_months','Movement','Gender']
    df = df[variables]

    for variable in variables[:3]:
        df[variable] = demean_variable(df[variable].values)

    stats_data = np.vstack([df[in_variable].values,
                  df['Age_in_months'].values,
                  df['Movement'].values,
                  df['Gender'].values,
                  np.ones(len(df))]).T

    print df.corr()

    file = open(filename,'w')
    file.write('/NumWaves ' + str(stats_data.shape[1]) + '\n')
    file.write('/NumPoints ' + str(stats_data.shape[0]) + '\n')
    file.write('/PPheights ' + ' '.join(map(str, np.max(stats_data,axis=0))) + '\n')
    file.write('/Matrix \n')
    [file.write(' '.join(map(str,stats_data[i,...])) + '\n') for i in range(0,stats_data.shape[0])];
    file.close()

def write_randomise_matrix(in_file,in_variable,filename):
    def demean_variable(variable):
        import numpy as np
        return variable - np.mean(variable)

    import pandas as pd 
    import numpy as np
    df = pd.read_csv(in_file)
    variables = [in_variable] + ['Movement','Gender']
    df = df[variables]

    for variable in variables[:3]:
        df[variable] = demean_variable(df[variable].values)

    stats_data = np.vstack([df[in_variable].values,
                  df['Movement'].values,
                  df['Gender'].values,
                  np.ones(len(df))]).T

    print df.corr()

    file = open(filename,'w')
    file.write('/NumWaves ' + str(stats_data.shape[1]) + '\n')
    file.write('/NumPoints ' + str(stats_data.shape[0]) + '\n')
    file.write('/PPheights ' + ' '.join(map(str, np.max(stats_data,axis=0))) + '\n')
    file.write('/Matrix \n')
    [file.write(' '.join(map(str,stats_data[i,...])) + '\n') for i in range(0,stats_data.shape[0])];
    file.close()

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
    file.write('1 0 0 0\n')
    file.write('-1 0 0 0\n')


def get_values_within_stats_mask(significance_image,value_image):
    import nibabel as nib
    import numpy as np
    mask_image = nib.load(significance_image)
    mask = mask_image.get_data()

    value_image = nib.load(value_image)
    all_data = value_image.get_data()

    # Getting the values within the mask
    means = list()

    for i in range(0,all_data.shape[-1]):
        data = all_data[...,i]
        means.append(np.mean(data[np.where(mask > 0.95)]))

    return means

def write_randomise_matrix_for_age(in_file,filename):
    def demean_variable(variable):
        import numpy as np
        return variable - np.mean(variable)

    import pandas as pd 
    import numpy as np
    df = pd.read_csv(in_file)
    variables = ['Age_in_months','Movement','Gender']
    df = df[variables]

    for variable in variables[:2]:
        df[variable] = demean_variable(df[variable].values)

    stats_data = np.vstack([df['Age_in_months'].values,
                  df['Movement'].values,
                  df['Gender'].values,
                  np.ones(len(df))]).T

    print df.corr()

    file = open(filename,'w')
    file.write('/NumWaves ' + str(stats_data.shape[1]) + '\n')
    file.write('/NumPoints ' + str(stats_data.shape[0]) + '\n')
    file.write('/PPheights ' + ' '.join(map(str, np.max(stats_data,axis=0))) + '\n')
    file.write('/Matrix \n')
    [file.write(' '.join(map(str,stats_data[i,...])) + '\n') for i in range(0,stats_data.shape[0])];
    file.close()