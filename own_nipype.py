from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, CommandLineInputSpec, CommandLine, traits, File, TraitedSpec
from nipype.interfaces.matlab import MatlabCommand

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


#==================================================================================================
# Moving tracts to a different space
class trk_CoregInputSpec(CommandLineInputSpec):
	in_file = File(exists=True, desc='whole-brain tractography in .trk format', 
		mandatory=True, position = 0, argstr="%s")
	output_file = File("coreg_tracks.trk", desc="whole-brain tractography in coregistered space", 
		position=1, argstr="%s", usedefault=True)
	FA_file = File(exists=True, desc='FA file in the same space as the .trk file', 
		mandatory=True, position = 2, argstr="-src %s")
	reference = File(exists=True, desc='Image that the .trk file will be registered to', 
		mandatory=True, position = 3, argstr="-ref %s")
	transfomation_matrix = File(exists=True, desc='FSL matrix with transform form original to new space', 
		mandatory=True, position = 4, argstr="-reg %s")

class trk_CoregOutputSpec(TraitedSpec):
	transformed_track_file = File(exists=True, desc="whole-brain tractography in new space")

class trk_Coreg(CommandLine):
	input_spec = trk_CoregInputSpec
	output_spec = trk_CoregOutputSpec

	_cmd = "track_transform"

	def _list_outputs(self):#
		import os 
		outputs = self.output_spec().get()
		outputs['transformed_track_file'] = os.path.abspath(self.inputs.output_file)
		return outputs

#==================================================================================================
# Extract b0
class Extractb0InputSpec(BaseInterfaceInputSpec):
	in_file = File(exists=True, desc='diffusion-weighted image (4D)', mandatory=True)

class Extractb0OutputSpec(TraitedSpec):
	out_file = File(exists=True, desc="First volume of the dwi file")

class Extractb0(BaseInterface):
	input_spec = Extractb0InputSpec
	output_spec = Extractb0OutputSpec

	def _run_interface(self, runtime):
		import nibabel as nib
		img = nib.load(self.inputs.in_file)
		data = img.get_data()
		affine = img.get_affine()

		from nipype.utils.filemanip import split_filename
		import os 
		outputs = self._outputs().get()
		fname = self.inputs.in_file
		_, base, _ = split_filename(fname)
		nib.save(nib.Nifti1Image(data[...,0],affine),os.path.abspath(base + '_b0.nii.gz'))
		return runtime

	def _list_outputs(self):
		from nipype.utils.filemanip import split_filename
		import os 
		outputs = self._outputs().get()
		fname = self.inputs.in_file
		_, base, _ = split_filename(fname)
		outputs["out_file"] = os.path.abspath(base + '_b0.nii.gz')
		return outputs
			
#==================================================================================================
# FA connectome construction

class FAconnectomeInputSpec(BaseInterfaceInputSpec):
	trackfile = File(exists=True, desc='whole-brain tractography in .trk format', mandatory=True)
	ROI_file = File(exists=True, desc='image containing the ROIs', mandatory=True)
	FA_file = File(exists=True, desc='fractional anisotropy map in the same soace as the track file', mandatory=True)
	output_file = File("FA_matrix.txt", desc="Adjacency matrix of ROIs with FA as conenction weight", usedefault=True)

class FAconnectomeOutputSpec(TraitedSpec):
	out_file = File(exists=True, desc="connectivity matrix of FA between each pair of ROIs")

class FAconnectome(BaseInterface):
	input_spec = FAconnectomeInputSpec
	output_spec = FAconnectomeOutputSpec

	def _run_interface(self, runtime):
		# Loading the ROI file
	    import nibabel as nib
	    import numpy as np
	    from dipy.tracking import utils 

	    img = nib.load(self.inputs.ROI_file)
	    data = img.get_data()
	    affine = img.get_affine()

	    # Getting the FA file
	    img = nib.load(self.inputs.FA_file)
	    FA_data = img.get_data()
	    FA_affine = img.get_affine()

	    # Loading the streamlines
	    from nibabel import trackvis
	    streams, hdr = trackvis.read(self.inputs.trackfile,points_space='rasmm')
	    streamlines = [s[0] for s in streams]
	    streamlines_affine = trackvis.aff_from_hdr(hdr,atleast_v2=True)

	    # Checking for negative values
	    from dipy.tracking._utils import _mapping_to_voxel, _to_voxel_coordinates
	    endpoints = [sl[0::len(sl)-1] for sl in streamlines]
	    lin_T, offset = _mapping_to_voxel(affine, (1.,1.,1.))
	    inds = np.dot(endpoints, lin_T)
	    inds += offset
	    negative_values = np.where(inds <0)[0]
	    for negative_value in sorted(negative_values, reverse=True):
			del streamlines[negative_value]

	    # Constructing the streamlines matrix
	    matrix,mapping = utils.connectivity_matrix(streamlines=streamlines,label_volume=data,affine=streamlines_affine,symmetric=True,return_mapping=True,mapping_as_streamlines=True)
	    matrix[matrix < 10] = 0

	    # Constructing the FA matrix
	    dimensions = matrix.shape
	    FA_matrix = np.empty(shape=dimensions)

	    for i in range(0,dimensions[0]):
	        for j in range(0,dimensions[1]):
	            if matrix[i,j]:
	                dm = utils.density_map(mapping[i,j], FA_data.shape, affine=streamlines_affine)
            		FA_matrix[i,j] = np.mean(FA_data[dm>0])
	            else:
	                FA_matrix[i,j] = 0

	    FA_matrix[np.tril_indices(n=len(FA_matrix))] = 0
	    FA_matrix = FA_matrix.T + FA_matrix - np.diagonal(FA_matrix)

	    from nipype.utils.filemanip import split_filename
	    _, base, _ = split_filename(self.inputs.trackfile)
	    np.savetxt(base + '_FA_matrix.txt',FA_matrix,delimiter='\t')
	    return runtime

	def _list_outputs(self):
		from nipype.utils.filemanip import split_filename
		import os 
		outputs = self._outputs().get()
		fname = self.inputs.trackfile
		_, base, _ = split_filename(fname)
		outputs["out_file"] = os.path.abspath(base + '_FA_matrix.txt')
		return outputs


#==================================================================================================
# Convert an adjacency matrix in txt format to NetworkX pck format

class TXT2PCKInputSpec(BaseInterfaceInputSpec):
	in_file = File(exists=True, desc='adjacency matrix in txt format', mandatory=True)

class TXT2PCKOutputSpec(TraitedSpec):
	out_file = File(exists=True, desc="NetworkX file in pck format")

class TXT2PCK(BaseInterface):
	input_spec = TXT2PCKInputSpec
	output_spec = TXT2PCKOutputSpec

	def _run_interface(self, runtime):
		# Reading the matrix file
		import numpy as np
		import networkx as nx

		adjacency_matrix = np.loadtxt(self.inputs.in_file)
		G = nx.from_numpy_matrix(adjacency_matrix)

		from nipype.utils.filemanip import split_filename
		_, base, _ = split_filename(self.inputs.in_file)
		nx.write_gpickle(G,path=base + '.pck')
		return runtime

	def _list_outputs(self):
		from nipype.utils.filemanip import split_filename
		import os 
		outputs = self._outputs().get()
		fname = self.inputs.in_file
		_, base, _ = split_filename(fname)
		outputs["out_file"] = os.path.abspath(base + '.pck')
		return outputs

#==================================================================================================
# Calling fsl_anat on T1 files
class FSLANATInputSpec(BaseInterfaceInputSpec):
	in_file = File(exists=True, desc='input structural image', mandatory=True)
	out_directory = File(exist=True, desc="output directory")

class FSLANATOutputSpec(TraitedSpec):
	fsl_anat_directory = traits.Directory(exists=True, desc="folder with processed T1 files")

class FSLANAT(BaseInterface):
	input_spec = FSLANATInputSpec
	output_spec = FSLANATOutputSpec

	def _run_interface(self, runtime):
		from subprocess import call
		subject = self.inputs.in_file.split('/')[-1].split('.')[0].split('_')[0]
		cmd = "fsl_anat --noreg --nononlinreg --noseg --nosubcortseg -i " + self.inputs.in_file + ' -o ' + self.inputs.out_directory + subject
		call(cmd,shell=True)

		return runtime

	def _list_outputs(self):
		import os 
		outputs = self.output_spec().get()
		subject = self.inputs.in_file.split('/')[-1].split('.')[0].split('_')[0]
		outputs['fsl_anat_directory'] = os.path.abspath(self.inputs.out_directory + subject + '.anat/')
		return outputs

#==================================================================================================
# Wavelet Despiking
import os
from string import Template

class WaveletDespikeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    subject_id = traits.String(mandatory=True)
    out_folder = File(mandatory=True)

class WaveletDespikeOutputSpec(TraitedSpec):
    out_folder = File(exists=True)

class WaveletDespike(BaseInterface):
    input_spec = WaveletDespikeInputSpec
    output_spec = WaveletDespikeOutputSpec

    def _run_interface(self, runtime):
        d = dict(in_file=self.inputs.in_file,
        out_folder=self.inputs.out_folder,
        subject_id=self.inputs.subject_id)
        #this is your MATLAB code template
        script = Template("""
        	cd '$out_folder'
        	WaveletDespike('$in_file','$subject_id')""").substitute(d)

        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_folder)
        return outputs