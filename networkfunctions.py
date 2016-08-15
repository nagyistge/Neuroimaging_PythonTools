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