import nibabel as nib
import numpy as np
import pandas as pd
import os,re

files = ['/imaging/jb07/CALM/DWI/NatbrainAtlas/Perisylvian/Anterior_Segment_Left.nii','/imaging/jb07/CALM/DWI/NatbrainAtlas/Perisylvian/Anterior_Segment_Right.nii','/imaging/jb07/CALM/DWI/NatbrainAtlas/Perisylvian/Long_Segment_Left.nii','/imaging/jb07/CALM/DWI/NatbrainAtlas/Perisylvian/Long_Segment_Right.nii','/imaging/jb07/CALM/DWI/NatbrainAtlas/Perisylvian/Posterior_Segment_Left.nii','/imaging/jb07/CALM/DWI/NatbrainAtlas/Perisylvian/Posterior_Segment_Right.nii']

for afile in files:

	from nipype.interfaces.fsl import FLIRT
	flirt = FLIRT()
	flirt.inputs.in_file = afile
	flirt.inputs.reference = '/imaging/local/software/fsl/v5.0.9/x86_64/fsl/data/standard/MNI152_T1_1mm.nii.gz'
	flirt.inputs.out_matrix_file = '/imaging/jb07/CALM/DWI/NatbrainAtlas/Perisylvian/00Average_Brain_MNI.mat'
	flirt.inputs.out_file = '/imaging/jb07/CALM/DWI/NatbrainAtlas/Perisylvian/00Average_Brain_MNI.nii.gz'
	flirt.inputs.dof = 6
	flirt.run()

	# Getting the images into proper MNI152 space
	from nipype.interfaces.fsl import FLIRT
	flirt = FLIRT()
	flirt.inputs.in_file = afile
	flirt.inputs.reference = '/imaging/local/software/fsl/v5.0.9/x86_64/fsl/data/standard/MNI152_T1_1mm.nii.gz'
	flirt.inputs.in_matrix_file = '/imaging/jb07/CALM/DWI/NatbrainAtlas/Perisylvian/00Average_Brain_MNI.mat'
	flirt.inputs.apply_xfm = True
	flirt.inputs.out_file = afile.split('.')[0] + '_MNI.nii.gz'
	flirt.run()


original_data = os.listdir('/imaging/jb07/WorkingMemory_Analysis/analysis_Mar2016/TBSS/origdata/')
IDs = list()

for original_file in original_data:
    if re.search('.nii.gz',original_file):
        IDs.append(original_file.split('.')[0])

# Getting the FA files in FMRIB space
all_FA = nib.load('/imaging/jb07/WorkingMemory_Analysis/analysis_Mar2016/TBSS/stats/all_FA.nii.gz')
affine = all_FA.get_affine()
all_FA = all_FA.get_data()

df = pd.DataFrame(index=range(0,len(all_FA[1,1,1,:])),columns=['Anterior_Segment_Left','Anterior_Segment_Right','Long_Segment_Left','Long_Segment_Right','Posterior_Segment_Left','Posterior_Segment_Right','ID'])


for tract in ['Anterior_Segment_Left','Anterior_Segment_Right','Long_Segment_Left','Long_Segment_Right','Posterior_Segment_Left','Posterior_Segment_Right']:
    img = nib.load('/imaging/jb07/CALM/DWI/NatbrainAtlas/Perisylvian/' + tract + '_MNI.nii.gz')
    atlas = img.get_data()
    atlas[np.where(atlas>0)] = 1

    for participant in range(0,all_FA.shape[3]):
        FA = all_FA[...,participant]
        df.ix[participant][tract] = np.mean(FA[atlas==1])
        df.ix[participant]['ID'] = IDs[participant]

df.to_csv('Perisylvian_atlas_results.csv')

