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

    # ==================================================================
    # Defining additional nodes

    from nipype.interfaces.base import BaseInterface
    from nipype.interfaces.base import BaseInterfaceInputSpec
    from nipype.interfaces.base import File
    from nipype.interfaces.base import TraitedSpec

    """
    This function calculates additional DTI measures, i.e. AD and RD, that
    FSL dtifi does not automatically generate
    """

    class AdditionalDTIMeasuresInputSpec(BaseInterfaceInputSpec):
        L1 = File(exists=True, desc='First eigenvalue image', mandatory=True)
        L2 = File(exists=True, desc='Second eigenvalue image', mandatory=True)
        L3 = File(exists=True, desc='Third eigenvalue image', mandatory=True)


    class AdditionalDTIMeasuresOutputSpec(TraitedSpec):
        AD = File(exists=True, desc="axial diffusivity (AD) image")
        RD = File(exists=True, desc="radial diffusivity (RD) image")


    class AdditionalDTIMeasures(BaseInterface):
        input_spec = AdditionalDTIMeasuresInputSpec
        output_spec = AdditionalDTIMeasuresOutputSpec

        def _run_interface(self, runtime):
            import nibabel as nib
            from nipype.utils.filemanip import split_filename

            L1 = nib.load(self.inputs.L1).get_data()
            L2 = nib.load(self.inputs.L2).get_data()
            L3 = nib.load(self.inputs.L3).get_data()
            affine = nib.load(self.inputs.L1).get_affine()

            RD = (L2 + L3) / 2

            fname = self.inputs.L1
            _, base, _ = split_filename(fname)
            nib.save(nib.Nifti1Image(L1, affine), base + '_AD.nii.gz')
            nib.save(nib.Nifti1Image(RD, affine), base + '_RD.nii.gz')
            return runtime

        def _list_outputs(self):
            from nipype.utils.filemanip import split_filename
            import os
            outputs = self._outputs().get()
            fname = self.inputs.L1
            _, base, _ = split_filename(fname)
            outputs["AD"] = os.path.abspath(base + '_AD.nii.gz')
            outputs["RD"] = os.path.abspath(base + '_RD.nii.gz')
            return outputs


    # ==================================================================
    """
    Denoising with non-local means
    This function is based on the example in the Dipy preprocessing tutorial:
    http://nipy.org/dipy/examples_built/denoise_nlmeans.html#example-denoise-nlmeans
    """

    class DipyDenoiseInputSpec(BaseInterfaceInputSpec):
        in_file = File(
            exists=True, desc='diffusion weighted volume for denoising', mandatory=True)


    class DipyDenoiseOutputSpec(TraitedSpec):
        out_file = File(exists=True, desc="denoised diffusion-weighted volume")


    class DipyDenoise(BaseInterface):
        input_spec = DipyDenoiseInputSpec
        output_spec = DipyDenoiseOutputSpec

        def _run_interface(self, runtime):
            import nibabel as nib
            import numpy as np
            from dipy.denoise.nlmeans import nlmeans
            from nipype.utils.filemanip import split_filename

            fname = self.inputs.in_file
            img = nib.load(fname)
            data = img.get_data()
            affine = img.get_affine()
            mask = data[..., 0] > 80
            a = data.shape

            denoised_data = np.ndarray(shape=data.shape)
            for image in range(0, a[3]):
                print(str(image + 1) + '/' + str(a[3] + 1))
                dat = data[..., image]
                # Calculating the standard deviation of the noise
                sigma = np.std(dat[~mask])
                den = nlmeans(dat, sigma=sigma, mask=mask)
                denoised_data[:, :, :, image] = den

            _, base, _ = split_filename(fname)
            nib.save(nib.Nifti1Image(denoised_data, affine),
                     base + '_denoised.nii.gz')

            return runtime

        def _list_outputs(self):
            from nipype.utils.filemanip import split_filename
            import os
            outputs = self._outputs().get()
            fname = self.inputs.in_file
            _, base, _ = split_filename(fname)
            outputs["out_file"] = os.path.abspath(base + '_denoised.nii.gz')
            return outputs

    #==============================================================

    # Main pipeline

    #==============================================================
    # Loading required packages
    from nipype import SelectFiles
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
