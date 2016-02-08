# Neuroimaging_PythonTools
These files contain some workflows based on (NiPyPe)[http://www.mit.edu/~satra/nipype-nightly/documentation.html], (DiPy)[http://nipy.org/dipy/], and my own functions for processing diffusion-weighted MRI data. 

The 'own_nipype' modules contains nodes with extensions for NiPyPe that are used in these workflows. 

## Functions in dwi_workflows:
__dwi_preproc__: script for advanced pre-processing of diffusin-weighted volumes
__dwi_preproc_minimal__: script for standard preprocessing of diffusion-weighted volumes
__dwi_preproc_restore__: tensor fitting using the RESTORE algorithm for motion correction
__CSD_determininistic_tractography__: reconstruction of a CSD model with deterministic tracking for whole-brain tractography
__CSD_determininistic_tractography__: reconstruction of a CSD model with probabilistic tracking for whole-brain tractography


written by Joe Bathelt, PhD MSc  
MRC Cognition & Brain Sciences Unit  
Cambridge, UK 
