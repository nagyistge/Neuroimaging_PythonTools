from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, File, TraitedSpec

#==================================================================================================
# Denoising with non-local means
# This function is based on the example in the Dipy preprocessing tutorial:
# http://nipy.org/dipy/examples_built/denoise_nlmeans.html#example-denoise-nlmeans

class DipyDenoiseInputSpec(BaseInterfaceInputSpec):
	in_file = File(exists=True, desc='diffusion weighted volume for denoising', mandatory=True)

class DipyDenoiseOutputSpec(TraitedSpec):
	out_file = File(exists=True, desc="denoised diffusion-weighted volume")

class DipyDenoise(BaseInterface):
	input_spec = DipyDenoiseInputSpec
	output_spec = DipyDenoiseOutputSpec

	def _run_interface(self, runtime):
		import nibabel as nib
		import numpy as np
		import matplotlib.pyplot as plt
		from dipy.denoise.nlmeans import nlmeans
		from nipype.utils.filemanip import split_filename

		fname = self.inputs.in_file
		img = nib.load(fname)
		data = img.get_data()
		affine = img.get_affine()
		mask = data[..., 0] > 80
		a = data.shape 

		denoised_data = np.ndarray(shape=data.shape)
		for image in range(0,a[3]):
		    print(str(image + 1) + '/' + str(a[3] + 1))
		    dat = data[...,image]
		    sigma = np.std(dat[~mask]) # Calculating the standard deviation of the noise 
		    den = nlmeans(dat, sigma=sigma, mask=mask)
		    denoised_data[:,:,:,image] = den

		_, base, _ = split_filename(fname)
		nib.save(nib.Nifti1Image(denoised_data, affine), base + '_denoised.nii')

		return runtime

	def _list_outputs(self):
		from nipype.utils.filemanip import split_filename
		import os 
		outputs = self._outputs().get()
		fname = self.inputs.in_file
		_, base, _ = split_filename(fname)
		outputs["out_file"] = os.path.abspath(base + '_denoised.nii')
		return outputs

#==================================================================================================
# Fitting the Tensor models with RESTORE

class DipyRestoreInputSpec(BaseInterfaceInputSpec):
	in_file = File(exists=True, desc='diffusion weighted volume', mandatory=True)
	bval = File(exists=True, desc='FSL-style b-value file', mandatory=True)
	bvec = File(exists=True, desc='FSL-style b-vector file', mandatory=True)

class DipyRestoreOutputSpec(TraitedSpec):
	out_file = File(exists=True, desc="fitted FA file")


class DipyRestore(BaseInterface):
	input_spec = DipyRestoreInputSpec
	output_spec = DipyRestoreOutputSpec

	def _run_interface(self, runtime):
		import dipy.reconst.dti as dti
		import dipy.denoise.noise_estimate as ne
		from dipy.core.gradients import gradient_table
		from nipype.utils.filemanip import split_filename

		import nibabel as nib

		fname = self.inputs.in_file
		img = nib.load(fname)
		data = img.get_data()
		affine = img.get_affine()

		bvals = self.inputs.bval
		bvecs = self.inputs.bvec

		gtab = gradient_table(bvals, bvecs)
		sigma = ne.estimate_sigma(data)
		dti = dti.TensorModel(gtab,fit_method='RESTORE', sigma=sigma)
		dtifit = dti.fit(data)
		fa = dtifit.fa

		_, base, _ = split_filename(fname)
		nib.save(nib.Nifti1Image(fa, affine), base + '_FA.nii')

		return runtime

	def _list_outputs(self):
		import os 
		from nipype.utils.filemanip import split_filename

		outputs = self._outputs().get()
		fname = self.inputs.in_file
		_, base, _ = split_filename(fname)
		outputs["out_file"] = os.path.abspath(base + '_FA.nii')
		return outputs


#==================================================================================================
# Deterministic tracking based on the CSD model

class CSDdetInputSpec(BaseInterfaceInputSpec):
	in_file = File(exists=True, desc='diffusion weighted volume', mandatory=True)
	bval = File(exists=True, desc='FSL-style b-value file', mandatory=True)
	bvec = File(exists=True, desc='FSL-style b-vector file', mandatory=True)
	FA_file = File(exists=True, desc='FA map', mandatory=True)
	brain_mask = File(exists=True, desc='FA map', mandatory=True)

class CSDdetOutputSpec(TraitedSpec):
	out_file = File(exists=True, desc="streamline trackfile")	

class CSDdet(BaseInterface):
	input_spec = CSDdetInputSpec
	output_spec = CSDdetOutputSpec

	def _run_interface(self, runtime):
		import numpy as np
		import nibabel as nib
		from dipy.io import read_bvals_bvecs
		from dipy.core.gradients import gradient_table
		from nipype.utils.filemanip import split_filename

		# Loading the data
		fname = self.inputs.in_file
		img = nib.load(fname)
		data = img.get_data()
		affine = img.get_affine()

		FA_fname = self.inputs.FA_file
		FA_img = nib.load(FA_fname)
		fa = FA_img.get_data()
		affine = FA_img.get_affine()
		affine = np.matrix.round(affine)

		mask_fname = self.inputs.brain_mask
		mask_img = nib.load(mask_fname)
		mask = mask_img.get_data()

		bval_fname = self.inputs.bval
		bvals = np.loadtxt(bval_fname)

		bvec_fname = self.inputs.bvec
		bvecs = np.loadtxt(bvec_fname)
		bvecs = np.vstack([bvecs[0,:],bvecs[1,:],bvecs[2,:]]).T
		gtab = gradient_table(bvals, bvecs)

		# Creating a white matter mask
		fa = fa*mask
		white_matter = fa >= 0.2

		# Creating a seed mask
		from dipy.tracking import utils
		seeds = utils.seeds_from_mask(white_matter, density=[2, 2, 2], affine=affine)

		# Fitting the CSA model
		from dipy.reconst.shm import CsaOdfModel
		from dipy.data import default_sphere
		from dipy.direction import peaks_from_model
		csa_model = CsaOdfModel(gtab, sh_order=8)
		csa_peaks = peaks_from_model(csa_model, data, default_sphere,
		                             relative_peak_threshold=.8,
		                             min_separation_angle=45,
		                             mask=white_matter)

		from dipy.tracking.local import ThresholdTissueClassifier
		classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)

		# CSD model
		from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel, auto_response)
		response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
		csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
		csd_fit = csd_model.fit(data, mask=white_matter)

		from dipy.direction import DeterministicMaximumDirectionGetter
		det_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
		                                                    max_angle=45.,
		                                                    sphere=default_sphere)

		# Tracking
		from dipy.tracking.local import LocalTracking
		streamlines = LocalTracking(det_dg, classifier, seeds, affine,
		                            step_size=.5, maxlen=200, max_cross=1)

		# Compute streamlines and store as a list.
		streamlines = list(streamlines)

		# Saving the trackfile
		from dipy.io.trackvis import save_trk
		_, base, _ = split_filename(fname)
		save_trk(base + '_CSDdet.trk', streamlines, affine, fa.shape)

		return runtime

	def _list_outputs(self):
		from nipype.utils.filemanip import split_filename
		import os 

		outputs = self._outputs().get()
		fname = self.inputs.in_file
		_, base, _ = split_filename(fname)
		outputs["out_file"] = os.path.abspath(base + '_CSDdet.trk')
		return outputs

#==================================================================================================
# Probabilistic tracking based on the CSD model

class CSDprobInputSpec(BaseInterfaceInputSpec):
	in_file = File(exists=True, desc='diffusion weighted volume', mandatory=True)
	bval = File(exists=True, desc='FSL-style b-value file', mandatory=True)
	bvec = File(exists=True, desc='FSL-style b-vector file', mandatory=True)
	FA_file = File(exists=True, desc='FA map', mandatory=True)
	brain_mask = File(exists=True, desc='FA map', mandatory=True)

class CSDprobOutputSpec(TraitedSpec):
	out_file = File(exists=True, desc="streamline trackfile")	

class CSDprob(BaseInterface):
	input_spec = CSDdetInputSpec
	output_spec = CSDdetOutputSpec

	def _run_interface(self, runtime):
		import numpy as np
		import nibabel as nib
		from dipy.io import read_bvals_bvecs
		from dipy.core.gradients import gradient_table
		from nipype.utils.filemanip import split_filename

		# Loading the data
		fname = self.inputs.in_file
		img = nib.load(fname)
		data = img.get_data()
		affine = img.get_affine()

		FA_fname = self.inputs.FA_file
		FA_img = nib.load(FA_fname)
		fa = FA_img.get_data()
		affine = FA_img.get_affine()
		affine = np.matrix.round(affine)

		mask_fname = self.inputs.brain_mask
		mask_img = nib.load(mask_fname)
		mask = mask_img.get_data()

		bval_fname = self.inputs.bval
		bvals = np.loadtxt(bval_fname)

		bvec_fname = self.inputs.bvec
		bvecs = np.loadtxt(bvec_fname)
		bvecs = np.vstack([bvecs[0,:],bvecs[1,:],bvecs[2,:]]).T
		gtab = gradient_table(bvals, bvecs)

		# Creating a white matter mask
		fa = fa*mask
		white_matter = fa >= 0.2

		# Creating a seed mask
		from dipy.tracking import utils
		seeds = utils.seeds_from_mask(white_matter, density=[2, 2, 2], affine=affine)

		# Fitting the CSA model
		from dipy.reconst.shm import CsaOdfModel
		from dipy.data import default_sphere
		from dipy.direction import peaks_from_model
		csa_model = CsaOdfModel(gtab, sh_order=8)
		csa_peaks = peaks_from_model(csa_model, data, default_sphere,
		                             relative_peak_threshold=.8,
		                             min_separation_angle=45,
		                             mask=white_matter)

		from dipy.tracking.local import ThresholdTissueClassifier
		classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)

		# CSD model
		from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel, auto_response)
		response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
		csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
		csd_fit = csd_model.fit(data, mask=white_matter)

		from dipy.direction import ProbabilisticDirectionGetter
		prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
		                                                    max_angle=45.,
		                                                    sphere=default_sphere)

		# Tracking
		from dipy.tracking.local import LocalTracking
		streamlines = LocalTracking(prob_dg, classifier, seeds, affine,
		                            step_size=.5, maxlen=200, max_cross=1)

		# Compute streamlines and store as a list.
		streamlines = list(streamlines)

		# Saving the trackfile
		from dipy.io.trackvis import save_trk
		_, base, _ = split_filename(fname)
		save_trk(base + '_CSDprob.trk', streamlines, affine, fa.shape)

		return runtime

	def _list_outputs(self):
		from nipype.utils.filemanip import split_filename
		import os 

		outputs = self._outputs().get()
		fname = self.inputs.in_file
		_, base, _ = split_filename(fname)
		outputs["out_file"] = os.path.abspath(base + '_CSDprob.trk')
		return outputs


from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, File, TraitedSpec

#==================================================================================================
# FA connectome construction

class DipyDenoiseInputSpec(BaseInterfaceInputSpec):
	trackfile = File(exists=True, desc='whole-brain tractography in .trk format', mandatory=True)
	ROI_file = File(exists=True, desc='image containing the ROIs', mandatory=True)
	FA_file = File(exists=True, desc='fractional anisotropy map in the same soace as the track file', mandatory=True)
	FA2structural_matrix = File(exists=True, desc='FSL transformation matrix that maps from native to common space', mandatory=True)

class DipyDenoiseOutputSpec(TraitedSpec):
	out_file = File(exists=True, desc="connectivity matrix of FA between each pair of ROIs")

class DipyDenoise(BaseInterface):
	input_spec = DipyDenoiseInputSpec
	output_spec = DipyDenoiseOutputSpec

	def _run_interface(self, runtime):
		# Loading the ROI file
	    import nibabel as nib
	    import numpy as np
	    from dipy.tracking import utils 

	    img = nib.load(ROI_file)
	    data = img.get_data()
	    affine = img.get_affine()

	    # Getting the FA file
	    img = nib.load(FA_file)
	    FA_data = img.get_data()
	    FA_affine = img.get_affine()

	    # Loading the streamlines
	    from nibabel import trackvis
	    streams, hdr = trackvis.read(trackfile,points_space='rasmm')
	    streamlines = [s[0] for s in streams]
	    streamlines_affine = trackvis.aff_from_hdr(hdr,atleast_v2=True)

	    # Constructing the streamlines matrix
	    matrix,mapping = utils.connectivity_matrix(streamlines=streamlines,label_volume=data,affine=streamlines_affine,symmetric=True,return_mapping=True,mapping_as_streamlines=True)

	    # Constructing the FA matrix
	    dimensions = matrix.shape
	    FA_matrix = np.empty(shape=dimensions)

	    for i in range(0,dimensions[0]):
	        for j in range(0,dimensions[1]):
	            if matrix[i,j]:
	                dm = utils.density_map(mapping[i,j], data.shape, affine=streamlines_affine)
	                FA_matrix[i,j] = np.mean(FA_data[dm>0])
	            else:
	                FA_matrix[i,j] = 0

	    FA_matrix[np.tril_indices(n=len(FA_matrix))] = 0
	    FA_matrix = FA_matrix.T + FA_matrix - np.diagonal(FA_matrix)
	    np.savetxt(base + '_FA_matrix.txt',delimiter='\t')
	    return runtime

	def _list_outputs(self):
		from nipype.utils.filemanip import split_filename
		import os 
		outputs = self._outputs().get()
		fname = self.inputs.in_file
		_, base, _ = split_filename(fname)
		outputs["out_file"] = os.path.abspath(base + '_FA_matrix.txt')
		return outputs