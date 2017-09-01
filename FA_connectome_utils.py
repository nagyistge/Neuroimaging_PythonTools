
def FA_connectome(trackfile,ROI_file,FA_file,FA2structural_matrix):
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

    return FA_matrix

def trk_MNI_coregistration(trackfile,FA_file,out_FA_file,out_FA_matrix,out_trackfile):
    # Co-registring the participant's data to the MNI template
    from nipype.interfaces import fsl
    flt = fsl.FLIRT(dof=12, cost_func='corratio')
    flt.inputs.in_file = FA_file
    flt.inputs.reference = '/imaging/jb07/ZDHHC9/FreeSurfer/fsaverage/mri/brain.nii.gz'
    flt.inputs.out_file = out_FA_file
    flt.inputs.out_matrix_file = out_FA_matrix
    flt.run()
    
    # Moving the whole-brain streamlines from participant space to MNI space using the transformation matrix from step 1
    from subprocess import call 
    command = 'track_transform ' + trackfile + ' -src ' + FA_file + ' -ref ' + reference + ' -reg ' + flt.inputs.out_matrix_file + " -reg_type 'flirt' " + out_trackfile
    call(command,shell=True)
    
def get_parcellation_labels_from_LUT(parcellation_lookup_table):

    #================================
    # Parcellation labels
    # This function get the labels corresponding to the values in the look-up table. 
    # This function expects a FreeSurfer-style lookup table, i.e. label numbers in the first column,
    # corresponding labels in the second column
    #================================

    """
    inputs:    
    parcellation_lookup_table: filename of the look-up table

    outputs:
    pandas dataframe containing the label number and the corresponding label
    """
    # Getting the labels and order
    import pandas as pd
    import re

    numbers = list()
    labels = list()

    file = open(parcellation_lookup_table,'r')

    for line in file:
        if len(line) > 4 and not re.search('#',line):
            content = list()

            entries = line.split(' ')
            for entry in entries:
                if entry:
                    content.append(entry)

            if len(content[0]) > 3:
                parts = content[0].split('\t')
                content[0] = parts[0]
                content[1] = parts[1]

            numbers.append(content[0])
            labels.append(content[1])

    return pd.Series(labels,index=numbers,name='label')

def get_parcellation_labels_from_gpickle(label_dict):

    #================================
    # Parcellation labels
    # This function reads the labels of ROIs from a gpickle dictionary
    #================================

    """ 
    inputs:
    label_dict: gpickle dictionary containing the ROI information

    outputs:
    pandas series containing the labels and index numbers of all ROIs
    """

    import pandas as pd
    import networkx as nx
    
    numbers = list()
    labels = list()

    Labels = nx.read_gpickle(label_dict)
    for label in Labels:
        numbers.append(label)
        labels.append(Labels[label]['labels'])
    
    return pd.Series(labels,index=numbers,name='label')

def remove_non_cortical_ROIs(labels,adjmat):

    #================================
    # Remove non-cortical label from a list of parcellation labels
    # This function removes all non-cortical ROIs from a list of ROIs. 
    # The cortical ROIs have to be labelled with 'ctx-' to be recognised.
    # The corresponding entries are also removed from the adjacency matrix.
    #================================

    """
    inputs:
    labels: list of labels
    adjmat: adjacency_matrix in the same order as the labels

    outputs:
    list of labels without non-cortical ROIs
    adjacency matrix without entries for non-cortical ROIs
    """

    import numpy as np
    import re 
    
    counter = 0
    non_cortical_labels = list()

    for label in labels:
        if not re.search('ctx',label):
            non_cortical_labels.append(counter)

        counter += 1

    labels = np.delete(labels,non_cortical_labels)
    adjmat = np.delete(adjmat,non_cortical_labels,axis=0)
    adjmat = np.delete(adjmat,non_cortical_labels,axis=1)
    
    return (labels,adjmat)

def get_label_order_from_file(label_order_file):
    #================================
    # Get labels from a saved file
    # This function reads a label file and return the labels in the same order as a list
    # This may be useful for re-order labels according to an external file
    #================================

    """ 
    inputs:
    label_order_file: text file containing the labels

    outputs:
    list of labels in order of appearance in the text file
    """

    import pandas as pd
    new_order = pd.read_csv(label_order_file,names=['Label'])
    new_order = new_order['Label'].values.tolist()
    return new_order


def plot_circular_connectivity(adjacency_matrix,filename,colour_minimum,colour_maximum,labels,node_order,modules):
    #================================
    # Get labels from a saved file
    # Plot the connectome in a circular layout 
    #================================

    """ 
    inputs:
    adjacency_matrix: adjacency_matrix of the connectome
    filename: name for the output image
    colour_minimum: minimum of the colormap
    colour_maximum: maximum of the colormap
    labels: labels for the ROIs
    node_order: desired order of labels (if no re-ordering is necessary, pass labels)
    modules: assignment of nodes to modules (optional)

    outputs:
    image file showing the connectome in circular layout
    """
    from mne.viz import circular_layout, plot_connectivity_circle

    node_angles = circular_layout(labels,node_order,start_pos=90,group_boundaries=[0, len(lab) / 2])

    fig = plt.figure(num=None, figsize=(10, 12), dpi=300, facecolor='black')
    if modules:
        # Mapping the data to a colour map
        import matplotlib as mpl
        import matplotlib.cm as cm
        norm = mpl.colors.Normalize(vmin=min(modules), vmax=max(modules))
        cmap = cm.prism
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        colormap = m.to_rgba(modules)
        colormap = colormap[:,0:3]

        plot_connectivity_circle(adjacency_matrix,new_order,node_angles=node_angles,node_colors=colormap.tolist(),vmin=minimum,vmax=maximum,fig=fig)

    else:
        plot_connectivity_circle(adjacency_matrix,new_order,node_angles=node_angles,node_colors='w',vmin=minimum,vmax=maximum,fig=fig)

    fig.savefig(filename, facecolor='black')


