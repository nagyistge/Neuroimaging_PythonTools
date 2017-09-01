# Graph Theory function
def get_consensus_module_assignment(network, iterations):
    #================================
    # Obtain the consensus module structure
    #================================
    """
    inputs:
    adjacency_matrix: adjacency_matrix
    gamma: gamma value

    outputs:
    vector of module assignment for each node
    """

    import numpy as np
    consensus_matrices = list()

    for i in range(0,iterations):
        consensus_matrix,modules,q = get_consensus_matrix(network)
        consensus_matrices.append(consensus_matrix)

    mean_consensus_matrix = np.mean(consensus_matrices,axis=0)

    consensus_matrix,modules,q = get_consensus_matrix(mean_consensus_matrix)
    consensus_matrix2,modules,q = get_consensus_matrix(mean_consensus_matrix)

    while abs(np.sum(consensus_matrix - consensus_matrix2)) != 0:
        consensus_matrix,modules,q = get_consensus_matrix(mean_consensus_matrix)
        consensus_matrix2,modules,q = get_consensus_matrix(mean_consensus_matrix)

    return (modules, q)

def get_consensus_matrix(network):
    import bct
    import numpy as np
    modules,q = bct.modularity_louvain_und_sign(network, qtype='smp')
    module_matrix = np.repeat(modules,repeats=network.shape[0])
    module_matrix = np.reshape(module_matrix,newshape=network.shape)
    consensus_matrix = module_matrix == module_matrix.transpose()
    return (consensus_matrix.astype('float'), modules, q)

def consensus_thresholding(in_matrices, percentage_threshold):
    # ===============================================================#
    # Consensus thresholding of connectivity matrices
    # for method see: http://www.ncbi.nlm.nih.gov/pubmed/23296185
    # ===============================================================#
    """
    inputs:
    in_matrices: numpy array with connectivity matrices (dimensions: ROIs x ROIs x participants)
    percentage_threshold: ratio of connections to be retained, e.g. 0.6

    outputs:
    matrices only retained connections that occured in the percentage threshold of participants
    """

    import bct
    import numpy as np

    connection_consensus = np.sum(np.asarray(in_matrices > 0), 2).astype('float64')/in_matrices.shape[2]
    connection_consensus = bct.binarize(bct.threshold_absolute(W=connection_consensus, thr=percentage_threshold))
    consensus_matrices = in_matrices * np.reshape(np.repeat(connection_consensus,in_matrices.shape[2]), newshape=in_matrices.shape)

    return consensus_matrices

def plot_network(network):
    #================================
    # Plot an adjacency matrix of a network
    #================================
    """
    inputs:
    network: adjacency_matrix (NumPy array)

    outputs:
    matplotlib figure of adjancency matrix
    """

    import matplotlib.pyplot as plt

    plt.imshow(network,
               cmap='jet',
               interpolation='none',
               vmin=-1, vmax=1)
    cb = plt.colorbar()
    cb.set_label('Pearson correlation coefficient R')
    cb.ax.yaxis.set_label_position('left')

def plot_community_matrix(network, community_affiliation):
    #================================
    # Plot a community matrix
    #================================
    """
    inputs:
    network: adjacency_matrix (NumPy array)
    community_affiliation: array that indicates which community/module an node belongs to

    outputs:
    matplotlib figure of adjancency matrix order by modules, lines indicate community boundaries
    """

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    sorting_array = sorted(range(len(community_affiliation)), key=lambda k: community_affiliation[k])
    sorted_network = network[sorting_array, :]
    sorted_network = sorted_network[:, sorting_array]
    plt.imshow(sorted_network,
               cmap='jet',
               interpolation='none',
               vmin=0, vmax=1)

    ax = plt.gca()
    total_size = 0
    for community in np.unique(community_affiliation):
        size = sum(sorted(community_affiliation) == community)
        ax.add_patch( patches.Rectangle(
                (total_size, total_size),
                size,
                size,
                fill = False,
                edgecolor = 'w',
                alpha = None,
                linewidth = 1
            )
        )
        total_size += size

    ax.set_xticks(np.arange(0, len(network), 1))
    ax.set_yticks(np.arange(0, len(network), 1))
    ax.grid(linewidth=0.2, color='w', alpha=0.5)
    ax.set_xticklabels(' ')
    ax.set_yticklabels(' ')

    cb = plt.colorbar()
    cb.set_label('Pearson correlation coefficient R')
    cb.ax.yaxis.set_label_position('left')

def random_network_with_modules(size, number_of_modules, p_in, p_out):
    #================================
    # Generate a random network with modules
    #================================
    """
    inputs:
    size: number of nodes
    number_of_modules: number of modules
    p_in: connection probability within modules
    p_out: connection probability ouside modules

    outputs:
    random network with modular structure
    """

    import numpy as np

    module_size = round(float(size)/number_of_modules)
    matrix = np.zeros(shape=[size,size])

    for k in np.arange(0,number_of_modules,1):
        # positive connections
        for i in np.arange(0, module_size, 1):
            for j in np.arange(0, module_size, 1):
                if np.random.permutation(10)[0] < 10*p_in:   # Create connections with a certain probability
                    random_weight = np.random.uniform(low=0.1, high=1.0)
                    matrix[int(i+k*module_size),int(j+k*module_size)] = random_weight
                    matrix[int(j+k*module_size),int(i+k*module_size)] = random_weight


        # negative connections
        for i in np.arange((k+1)*module_size, len(matrix), 1):
            for j in np.arange(k*module_size, (k+1)*(module_size), 1):
                if np.random.permutation(10)[0] < 10*p_out:
                    random_weight = np.random.uniform(low=0.1, high=1.0)
                    matrix[int(i),int(j)] = random_weight
                    matrix[int(j),int(i)] = random_weight

    return matrix

def add_noise(network, percentage_noise):
    #================================
    # Add noise to an adjacency matrix
    #================================
    """
    inputs:
    network: adjacency_matrix (NumPy array)
    percentage_noise: percentage of Gaussian noise

    outputs:
    network adjacency matrix with added noise
    """

    import numpy as np

    network_wNoise = (1-(float(percentage_noise)/100))*network + (float(percentage_noise)/100)*np.random.normal(0, 1, network.shape)

    return network_wNoise

def get_connection_densities(network, community_affiliation):
    #================================
    # Get density of within and between module connections
    #================================
    """
    inputs:
    network: adjacency_matrix (NumPy array)
    community_affiliation: array that indicates which community/module an node belongs to

    outputs:
    density of connections within modules
    density of connections between modules
    """

    import networkx as nx
    import numpy as np

    network[network > 0] = 1. # binarizing the network

    G = nx.from_numpy_matrix(network) # original network
    for node in G.nodes():
         G.node[node]['community'] = community_affiliation[node]

    within_weights = list()
    between_weights = list()

    for edge in G.edges():
        if G.node[edge[0]]['community'] == G.node[edge[1]]['community']:
            within_weights.append(G.edge[edge[0]][edge[1]]['weight'])
        else:
            between_weights.append(G.edge[edge[0]][edge[1]]['weight'])

    connected_G = nx.from_numpy_matrix(np.ones(shape=network.shape)) # fully-connected network
    full_within_weights = list()
    full_between_weights = list()

    for node in connected_G.nodes():
         connected_G.node[node]['community'] = community_affiliation[node]

    for edge in connected_G.edges():
        if connected_G.node[edge[0]]['community'] == connected_G.node[edge[1]]['community']:
            full_within_weights.append(connected_G.edge[edge[0]][edge[1]]['weight'])
        else:
            full_between_weights.append(connected_G.edge[edge[0]][edge[1]]['weight'])

    within_density = sum(within_weights)/sum(full_within_weights)
    between_density = sum(between_weights)/sum(full_between_weights)

    return(within_density, between_density)

def aparc_indices(parcellation_file):
    """
    This function returns the indices of ROIs in FreeSurfer's aparc parcellation

    inputs:
        parcellation_file: Nifti file with the parcellation numbers

    outputs:
        indices to retain in order of appearance
    """

    import nibabel as nib
    import numpy as np

    regions_to_include = {'Left_Thalamus_Proper': 10,
                         'Left_Caudate': 11,
                         'Left_Putamen': 12,
                         'Left_Pallidum': 13,
                         'Lef_Hippocampus': 17,
                         'Left_Amygdala': 18,
                         'Left_Accumbens_area': 26,
                         'Right_Thalamus_proper': 49,
                         'Right_Caudate': 50,
                         'Right_Putamen': 51,
                         'Right_Pallidum': 52,
                         'Right_Hippocampus': 53,
                         'Right_Amygdala': 54,
                         'Right_Accumbens_area': 58,
                         'ctx-lh-bankssts': 1001,
                         'ctx-lh-caudalanteriorcingulate': 1002,
                         'ctx-lh-caudalmiddlefrontal': 1003,
                         'ctx-lh-cuneus': 1005,
                         'ctx-lh-entorhinal': 1006,
                         'ctx-lh-fusiform': 1007,
                         'ctx-lh-inferiorparietal': 1008,
                         'ctx-lh-inferiortemporal': 1009,
                         'ctx-lh-isthmuscingulate': 1010,
                         'ctx-lh-lateraloccipital': 1011,
                         'ctx-lh-lateralorbitofrontal': 1012,
                         'ctx-lh-lingual': 1013,
                         'ctx-lh-medialorbitofrontal': 1014,
                         'ctx-lh-middletemporal': 1015,
                         'ctx-lh-parahippocampal': 1016,
                         'ctx-lh-paracentral': 1017,
                         'ctx-lh-parsopercularis': 1018,
                         'ctx-lh-parsorbitalis': 1019,
                         'ctx-lh-parstriangularis': 1020,
                         'ctx-lh-pericalcarine': 1021,
                         'ctx-lh-postcentral': 1022,
                         'ctx-lh-posteriorcingulate': 1023,
                         'ctx-lh-precentral': 1024,
                         'ctx-lh-precuneus': 1025,
                         'ctx-lh-rostralanteriorcingulate': 1026,
                         'ctx-lh-rostralmiddlefrontal': 1027,
                         'ctx-lh-superiorfrontal': 1028,
                         'ctx-lh-superiorparietal': 1029,
                         'ctx-lh-superiortemporal': 1030,
                         'ctx-lh-supramarginal': 1031,
                         'ctx-lh-frontalpole': 1032,
                         'ctx-lh-temporalpole': 1033,
                         'ctx-lh-transversetemporal': 1034,
                         'ctx-lh-insula': 1035,
                         'ctx-rh-bankssts': 2001,
                         'ctx-rh-caudalanteriorcingulate': 2002,
                         'ctx-rh-caudalmiddlefrontal': 2003,
                         'ctx-rh-cuneus': 2005,
                         'ctx-rh-entorhinal': 2006,
                         'ctx-rh-fusiform': 2007,
                         'ctx-rh-inferiorparietal': 2008,
                         'ctx-rh-inferiortemporal': 2009,
                         'ctx-rh-isthmuscingulate': 2010,
                         'ctx-rh-lateraloccipital': 2011,
                         'ctx-rh-lateralorbitofrontal': 2012,
                         'ctx-rh-lingual': 2013,
                         'ctx-rh-medialorbitofrontal': 2014,
                         'ctx-rh-middletemporal': 2015,
                         'ctx-rh-parahippocampal': 2016,
                         'ctx-rh-paracentral': 2017,
                         'ctx-rh-parsopercularis': 2018,
                         'ctx-rh-parsorbitalis': 2019,
                         'ctx-rh-parstriangularis': 2020,
                         'ctx-rh-pericalcarine': 2021,
                         'ctx-rh-postcentral': 2022,
                         'ctx-rh-posteriorcingulate': 2023,
                         'ctx-rh-precentral': 2024,
                         'ctx-rh-precuneus': 2025,
                         'ctx-rh-rostralanteriorcingulate': 2026,
                         'ctx-rh-rostralmiddlefrontal': 2027,
                         'ctx-rh-superiorfrontal': 2028,
                         'ctx-rh-superiorparietal': 2029,
                         'ctx-rh-superiortemporal': 2030,
                         'ctx-rh-supramarginal': 2031,
                         'ctx-rh-frontalpole': 2032,
                         'ctx-rh-temporalpole': 2033,
                         'ctx-rh-transversetemporal': 2034,
                         'ctx-rh-insula': 2035}


    regions = np.unique(nib.load(parcellation_file).get_data())[1:]
    indices_to_retain = np.where(np.in1d(regions, regions_to_include.values()))

    return indices_to_retain
