## Graph Theory function
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

def plot_network(network):
    import matplotlib.pyplot as plt
    
    plt.imshow(network, 
               cmap='jet',
               interpolation='none',
               vmin=-1, vmax=1)
    cb = plt.colorbar()
    cb.set_label('Pearson correlation coefficient R')
    cb.ax.yaxis.set_label_position('left')

def plot_community_matrix(network, community_affiliation):
    import matplotlib.pyplot as plt

    sorting_array = sorted(range(len(community_affiliation)), key=lambda k: community_affiliation[k])
    sorted_network = network[sorting_array, :]
    sorted_network = sorted_network[:, sorting_array]
    plt.imshow(sorted_network, 
               cmap='jet',
               interpolation='none',
               vmin=-1, vmax=1)
    cb = plt.colorbar()
    cb.set_label('Pearson correlation coefficient R')
    cb.ax.yaxis.set_label_position('left')

def random_network_with_modules(size, number_of_modules, p_in, p_out):
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