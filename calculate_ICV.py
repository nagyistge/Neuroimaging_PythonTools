#! /usr/bin/env python
import optparse
import os
import re
import sys


# ======================================================================

def main():
    p = optparse.OptionParser()

    p.add_option('--input_folder', '-b')
    p.add_option('--subject_list', '-s')
    p.add_option('--out_folder', '-o')

    sys.path.append(os.path.realpath(__file__))

    options, arguments = p.parse_args()
    input_folder = options.input_folder
    out_folder = options.out_folder
    subject_list = options.subject_list
    subject_list = [subject for subject in subject_list.split(
        ',') if subject]

    # ==================================================================
    # Additional interfaces
    from nipype.interfaces.base import BaseInterface
    from nipype.interfaces.base import BaseInterfaceInputSpec
    from nipype.interfaces.base import File
    from nipype.interfaces.base import traits
    from nipype.interfaces.base import TraitedSpec

    class SIENAXInputSpec(BaseInterfaceInputSpec):
            in_file = File(desc = "Input volume", exists = True,
                                mandatory = True, position = 0, argstr="%s")
            bet_options = traits.String(desc = "options to pass to BET brain extraction (inside double-quotes), e.g. -B '-f 0.3'", mandatory = False, position = 1, argstr="-B %s")
            output_folder = traits.File(desc = "set output directory (default output is <input>_sienax)", mandatory = True, position = 2, argstr="-o %s")

    class SIENAXOutputSpec(TraitedSpec):
            report = File(desc = "sienax report", exists = True)

    class SIENAX(BaseInterface):
        input_spec = SIENAXInputSpec
        output_spec = SIENAXOutputSpec

        def _run_interface(self, runtime):
            from subprocess import call
            in_file = self.inputs.in_file
            bet_options = self.inputs.bet_options
            output_folder = self.inputs.output_folder

            string = 'sienax ' + in_file + \
            '-o ' + output_folder

            if bet_options:
                string = string + ' -B ' + bet_options

            call(string, shell=True)
            return runtime

        def _list_outputs(self):
            import os
            outputs = self.output_spec().get()
            outputs['report'] = os.path.abspath(self.inputs.output_folder + 'report.sienax')
            return outputs

    # ==================================================================
    # Main workflow
    # ==================================================================

    from nipype.interfaces import fsl
    import nipype.interfaces.utility as util
    import nipype.pipeline.engine as pe
    from nipype import SelectFiles


    # ==================================================================
    # Defining the nodes for the workflow

    # Getting the subject ID
    infosource = pe.Node(interface=util.IdentityInterface(
        fields=['subject_id']), name='infosource')
    infosource.iterables = ('subject_id', subject_list)

    # Getting the relevant diffusion-weighted data
    templates = dict(T1='{subject_id}/anat/{subject_id}_T1w.nii.gz')

    selectfiles = pe.Node(SelectFiles(templates),
                          name='selectfiles')
    selectfiles.inputs.base_directory = os.path.abspath(input_folder)

    # Cropping the image
    fslroi = pe.Node(interface=fsl.ExtractROI(), name='fslroi')
    fslroi.inputs.t_min = 0
    fslroi.inputs.t_size = 1

    # Creating the output folder name for sienax
    func = 'def make_foldername(out_folder, subject_id): return out_folder + "ICV/_subject_id_" + subject_id + "/sienax/" + subject_id'
    folder_name = pe.Node(interface = util.Function(input_names=['out_folder', 'subject_id'], output_names=['sineax_folder']), name='folder_name')
    folder_name.inputs.function_str = func
    folder_name.inputs.out_folder = out_folder

    # Running SIENAX
    sienax = pe.Node(interface=SIENAX(), name='sienax')

    # ==================================================================
    # Setting up the workflow
    ICV = pe.Workflow(name='ICV')

    # Reading in files
    ICV.connect(infosource, 'subject_id', selectfiles, 'subject_id')

    # Cropping the image
    ICV.connect(selectfiles, 'T1', fslroi, 'in_file')
    ICV.connect(infosource, 'subject_id', folder_name, 'subject_id')

    # Running SIENAX
    ICV.connect(fslroi, 'roi_file', sienax, 'in_file')
    ICV.connect(folder_name, 'sineax_folder', sienax, 'output_folder')

    # ===================================================================
    # Running the workflow
    ICV.base_dir = os.path.abspath(out_folder)
    ICV.write_graph()
    ICV.run()

if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())
