# Neuroimaging_PythonTools
These files contain some workflows based on (NiPyPe)[http://www.mit.edu/~satra/nipype-nightly/documentation.html], (DiPy)[http://nipy.org/dipy/], and my own functions for processing diffusion-weighted MRI data. 

The 'own_nipype' modules contains nodes with extensions for NiPyPe that are used in these workflows. 

## Functions in dwi_workflows:
__dwi_preproc__: script for advanced pre-processing of diffusin-weighted volumes  
__dwi_preproc_minimal__: script for standard preprocessing of diffusion-weighted volumes  
__dwi_preproc_restore__: tensor fitting using the RESTORE algorithm for motion correction  
__CSD_determininistic_tractography__: reconstruction of a CSD model with deterministic tracking for whole-brain tractography  
__CSD_determininistic_tractography__: reconstruction of a CSD model with probabilistic tracking for whole-brain tractography  
__DTI_calculate_RD__: function to calculate radial diffusivity  
__CSD_probabilistic_tractography_MRTrix__: reconstruction of the CSD model and probabilistic whole-brain tractography using MRTrix functions  
__FA_connectome__: calculate a structural connectome that expresses the connection weighte between two regions as FA  



written by Joe Bathelt, PhD MSc  
MRC Cognition & Brain Sciences Unit  
Cambridge, UK 
