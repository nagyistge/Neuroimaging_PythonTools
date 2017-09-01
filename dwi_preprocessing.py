def dwi_preproc(subject_list, base_directory, out_directory, index_file, acqparams):
    """
    This function implements the dwi preprocessing workflow. The function takes a list of subject IDs and their parent directory - the data is expected to be stored according to the Brain Imaging Data Structure (BIDS).
    It then performs the preprocessing steps: denoising with non-local means (http://nipy.org/dipy/examples_built/denoise_nlmeans.html), FSL eddy_correct to correct for eddy current and participant motion, resampling to 1mm isotropic resolution with trilinear interpolation, extraction of the first b0 volume, brain extraction with FSL bet, and fitting of the diffusion tensor model with FSL dtifit

    inputs:
    subject list: python list of string with the subject IDs
    base_directory: directory in which the raw data is stored (diffusion weighted volume, bval and bvecs file)
    out_directory: directory where is output will be stored

    written by Joe Bathelt,PhD
    MRC Cognition & Brain Sciences Unit
    joe.bathelt@mrc-cbu.cam.ac.uk
    """
    #==============================================================
    # Loading required packages
    from nipype import SelectFiles
    from additional_interfaces import AdditionalDTIMeasures
    from additional_interfaces import DipyDenoise
    import nipype.interfaces.fsl as fsl
    import nipype.interfaces.io as nio
    import nipype.pipeline.engine as pe
    import nipype.interfaces.utility as util
    import os

    # ==============================================================
    # Defining the nodes for the workflow

    # Getting the subject ID
    infosource  = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),name='infosource')
    infosource.iterables = ('subject_id', subject_list)

    # Getting the relevant diffusion-weighted data
    templates = dict(dwi='{subject_id}/dwi/{subject_id}_dwi.nii.gz',
                     bvec='{subject_id}/dwi/{subject_id}_dwi.bvec',
                     bval='{subject_id}/dwi/{subject_id}_dwi.bval')

    selectfiles = pe.Node(SelectFiles(templates),
                       name="selectfiles")
    selectfiles.inputs.base_directory = os.path.abspath(base_directory)

    # Extract b0 image
    fslroi = pe.Node(interface=fsl.ExtractROI(), name='extract_b0')
    fslroi.inputs.t_min = 0
    fslroi.inputs.t_size = 1

    # Create a brain mask
    bet = pe.Node(interface=fsl.BET(
        frac=0.3, robust=False, mask=True, no_output=False), name='bet')

    # Eddy-current and motion correction
    eddy = pe.Node(interface=fsl.epi.Eddy(args='-v'), name='eddy')
    eddy.inputs.in_acqp  = acqparams
    eddy.inputs.in_index = index_file

    # Denoising
    dwi_denoise = pe.Node(interface=DipyDenoise(), name='dwi_denoise')

    # Fitting the diffusion tensor model
    dtifit = pe.Node(interface=fsl.DTIFit(), name='dtifit')

    # Getting AD and RD
    get_rd = pe.Node(interface=AdditionalDTIMeasures(), name='get_rd')

    # ==============================================================
    # Setting up the workflow
    dwi_preproc = pe.Workflow(name='dwi_preproc')

    # Diffusion data
    # Preprocessing
    dwi_preproc.connect(infosource, 'subject_id', selectfiles, 'subject_id')
    dwi_preproc.connect(selectfiles, 'dwi', fslroi, 'in_file')

    dwi_preproc.connect(selectfiles, 'dwi', eddy, 'in_file')
    dwi_preproc.connect(selectfiles, 'bval', eddy, 'in_bval')
    dwi_preproc.connect(selectfiles, 'bvec', eddy, 'in_bvec')

    dwi_preproc.connect(fslroi, 'roi_file', bet, 'in_file')
    dwi_preproc.connect(bet, 'mask_file', eddy, 'in_mask')
    dwi_preproc.connect(eddy, 'out_corrected', dwi_denoise, 'in_file')

    # Calculate diffusion measures
    dwi_preproc.connect(dwi_denoise, 'out_file', dtifit, 'dwi')
    dwi_preproc.connect(bet, 'mask_file', dtifit, 'mask')
    dwi_preproc.connect(selectfiles, 'bval', dtifit, 'bvals')
    dwi_preproc.connect(selectfiles, 'bvec', dtifit, 'bvecs')

    dwi_preproc.connect(infosource, 'subject_id', dtifit, 'base_name')
    dwi_preproc.connect(dtifit, 'L1', get_rd, 'L1')
    dwi_preproc.connect(dtifit, 'L2', get_rd, 'L2')
    dwi_preproc.connect(dtifit, 'L3', get_rd, 'L3')

    # ==============================================================
    # Running the workflow
    dwi_preproc.base_dir = os.path.abspath(out_directory)
    dwi_preproc.write_graph()
    dwi_preproc.write_graph()
    dwi_preproc.run()
