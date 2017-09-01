from subprocess import call

base_directory = '/imaging/jb07/CALM/CALM_BIDS/'
subject_list = ['CBU150084']
template_directory = '/home/jb07/joe_python/GitHub/ZDHHC9_connectome/NKI/'
out_directory = '/imaging/jb07/CALM/tests/'
parcellation_directory = '/home/jb07/joe_python/GitHub/ZDHHC9_connectome/FreeSurfer_templates/'
acquisition_parameters = '/imaging/jb07/CALM/CALM_BIDS/acqparams.txt'
index_file = '/imaging/jb07/CALM/CALM_BIDS/index.txt'

script_folder = '/imaging/jb07/CALM/tests/'

for subject in subject_list:

    # Writing a python script
    cmd = "python /home/jb07/joe_python/GitHub/ZDHHC9_connectome/connectome_pipeline.py " + \
    " --base_directory " + base_directory + \
    " --subject_list " + subject + \
    " --template_directory " + template_directory + \
    " --out_directory " + out_directory + \
    " --parcellation_directory " + parcellation_directory + \
    " --acquisition_parameters " + acquisition_parameters + \
    " --index_file " + index_file
    file = open(script_folder + subject + '_connectome.sh', 'w')
    file.write(cmd)
    file.close()

    cmd = 'qsub ' + script_folder + subject + '_connectome.sh' + ' -l walltime=48:00:00'
    call(cmd, shell=True)
