## General graph preprocessing
def relative_threshold_graph(NetworkX_Graph, threshold, weight):
    import networkx as nx
    import numpy as np
    
    edges = NetworkX_Graph.edges()
    edge_weights = list()
    
    for edge in edges:
        edge_weights.append(NetworkX_Graph[edge[0]][edge[1]][weight])
        
    edge_weights = np.asarray(edge_weights)

    threshold = threshold*np.max(edge_weights)
    
    return threshold

def binarize_graph(NetworkX_Graph, threshold, weight):
    import networkx as nx
    import numpy as np
    from scipy import integrate
    
    thresholded_graph = nx.Graph()
    edges = NetworkX_Graph.edges()
    
    for edge in edges:
        edge_weight = NetworkX_Graph[edge[0]][edge[1]][weight]
        if edge_weight > threshold:
            thresholded_graph.add_edge(edge[0],edge[1], weight=1)
                        
    return thresholded_graph

def threshold_graph(NetworkX_Graph, threshold, weight):
    import networkx as nx
    import numpy as np
    from scipy import integrate
    
    thresholded_graph = nx.Graph()
    edges = NetworkX_Graph.edges()
    
    for edge in edges:
        edge_weight = NetworkX_Graph[edge[0]][edge[1]][weight]
        if edge_weight > threshold:
            thresholded_graph.add_edge(edge[0],edge[1], weight=edge_weight)
                        
    return thresholded_graph

def invert_edge_weight(NetworkX_Graph):
    edges = NetworkX_Graph.edges()
        
    for edge in edges:
        NetworkX_Graph[edge[0]][edge[1]]['relative_weight'] = 1. - float(NetworkX_Graph[edge[0]][edge[1]]['relative_weight'])
    
    return NetworkX_Graph

def get_degrees(G,weighting_factor):
    # function to calculate the degree of all nodes 
    import numpy as np
    import networkx as nx

    if weighting_factor:
        degrees = nx.degree(G,weight=weighting_factor)
    else:
         degrees = nx.degree(G)
  
    
    degree_array = list()
    
    for degree in degrees:
            degree_array.append(degrees[degree])
    
    degree_array = np.asarray(degree_array)
    return np.sort(degree_array)

def calculate_total_number_of_edges(NetworkX_Graph):
    # function to get the total weight of edges in a network
    import numpy as np
    
    edges = NetworkX_Graph.edges()
    total_weight = list()
    
    for edge in edges:
        total_weight.append(NetworkX_Graph[edge[0]][edge[1]]['number_of_fibers'])
    
    total_weight = np.asarray(total_weight)
    total_weight = np.sum(total_weight)
    
    return total_weight

def sum_edge_weight(NetworkX_Graph,weight):
    import numpy as np

    edge_weights = list()
    edges = NetworkX_Graph.edges()
        
    for edge in edges:
        edge_weights.append(NetworkX_Graph[edge[0]][edge[1]][weight])
        
    maximum_weight = np.sum(np.asarray(edge_weights))
    return maximum_weight

def calculate_relative_edge_weight(NetworkX_Graph):
    # Function to calculate the relative weight from total fibre number counts
    edges = NetworkX_Graph.edges()
    
    for edge in edges:
        #NetworkX_Graph[edge[0]][edge[1]]['relative_weight'] = float(NetworkX_Graph[edge[0]][edge[1]]['number_of_fibers'])/total_number
        NetworkX_Graph[edge[0]][edge[1]]['relative_weight'] = float(NetworkX_Graph[edge[0]][edge[1]]['number_of_fibers'])/sum_edge_weight(NetworkX_Graph,'number_of_fibers')

    return NetworkX_Graph

## Plotting functions
def plot_degree_distribution(patient_degrees,control_degrees,labels):
    import matplotlib.pyplot as plt
    import numpy as np
    #control_degrees = control_degrees[::-1]
    #patient_degrees = patient_degrees[::-1]
    
    length = control_degrees.shape
    ticks = np.linspace(0,length[0],length[0])
    
    control_mean = np.mean(control_degrees,axis=0)
    control_max = control_mean + np.std(control_degrees,axis=0)/np.sqrt(len(control_degrees))
    control_min = control_mean - np.std(control_degrees,axis=0)/np.sqrt(len(control_degrees))
    
    patient_mean = np.mean(patient_degrees,axis=0)
    patient_max = patient_mean + np.std(patient_degrees,axis=0)/np.sqrt(len(patient_degrees))
    patient_min = patient_mean - np.std(patient_degrees,axis=0)/np.sqrt(len(patient_degrees))

    plt.fill_between(ticks,control_min,control_max, interpolate=True, alpha=0.3, color='blue')
    plt.fill_between(ticks,patient_min,patient_max, interpolate=True, alpha=0.3, color='red')
    
    control = plt.plot(control_mean, color='blue',label=labels[0])
    patient = plt.plot(patient_mean, color='red',label=labels[1])
    plt.legend()
    plt.ylabel('Node degree')
    
    plt.show()

def plot_adjancey_matrix(participant,threshold):
	import networkx as nx
	import numpy as np
	import matplotlib.pylab as plt
	G = nx.read_gpickle('/home/jb07/Documents/connectome/' + str(participant) + '_connectome.pck')
	threshold = relative_threshold_graph(G,threshold,'number_of_fibers')
	G = binarize_graph(G,threshold,'number_of_fibers')
	print 'number of edges:' + str(len(G.edges()))
	adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.float64, weight='weight')
	plt.imshow(adjacency_matrix)
	plt.colorbar()
	plt.show()

## Graph theory measures
def global_efficiency(participants,threshold):

	import networkx as nx
   	import numpy as np
   	from brainx import metrics

   	results = list()
    
   	for participant in participants:
   	        G = nx.read_gpickle('/home/jb07/Documents/connectome/' +  participant + '_connectome.pck')

   	        if threshold:
   	            threshold = relative_threshold_graph(G,threshold)
   	            G = binarize_graph(G,threshold)
   	        results.append(np.mean(metrics.glob_efficiency(G)))
    
   	results = np.asarray(results)
   	return results


def maximum_betweenness_centrality(participants,threshold):
	import networkx as nx
   	import numpy as np

   	results = list()
    
   	for participant in participants:
   	    G = nx.read_gpickle('/home/jb07/Documents/connectome/' +  participant + '_connectome.pck')
   	    threshold = relative_threshold_graph(G,0.1)
   	    G = binarize_graph(G,threshold)
   	    centralities = nx.betweenness_centrality(G,weight='weight')
        
   	    all_centralities = list()
   		
        for i in centralities:
       	   	all_centralities.append(centralities[i])
       	all_centralities = np.asarray(all_centralities)
       	max_centrality = np.max(all_centralities)
        
       	results.append(max_centrality)
        
   	results = np.asarray(results)
   	return results


def number_of_components(participants,threshold):
    import networkx as nx
    import numpy as np

    results = list()
    
    for participant in participants:
        G = nx.read_gpickle('/home/jb07/Documents/connectome/' +  participant + '_connectome.pck')
        G = calculate_relative_edge_weight(G)
    	threshold = relative_threshold_graph(G,0.1,'relative_weight')
    	G = binarize_graph(G, threshold, 'relative_weight')
        number_of_comps = nx.number_connected_components(G)
        results.append(number_of_comps)
        
    results = np.asarray(results)
    return results

def average_shortest_path(participants,threshold):
    import networkx as nx
    import numpy as np
    short_path = list()
    
    for participant in participants:
        G = nx.read_gpickle('/home/jb07/Documents/connectome/' +  participant + '_connectome.pck')
        G = calculate_relative_edge_weight(G)
    	threshold = relative_threshold_graph(G,0.1,'relative_weight')
    	G = binarize_graph(G, threshold, 'relative_weight')
        subgraphs = list(nx.connected_component_subgraphs(G))
        main_graph = subgraphs[0]
        short_path.append(nx.average_shortest_path_length(main_graph, weight='weight'))
        
    short_path = np.asarray(short_path)
    return short_path

def avg_clustering_coefficient(participants,threshold):
    clust_coeffs = list()
    import networkx as nx
    import numpy as np

    for participants in participants:
        G = nx.read_gpickle('/home/jb07/Documents/connectome/' +  participants + '_connectome.pck')
        G = calculate_relative_edge_weight(G)
    	threshold = relative_threshold_graph(G,0.1,'number_of_fibers')
    	G = binarize_graph(G, threshold, 'number_of_fibers')
        clust_coeffs.append(nx.average_clustering(G, count_zeros=True, weight='weight'))
          
    clust_coeffs = np.asarray(clust_coeffs)
    return clust_coeffs
