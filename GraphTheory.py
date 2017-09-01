def calculate_graph_measures(in_file):
	import numpy as np
	import networkx as nx
	adjacency_matrix = np.loadtxt(in_file)
	G = nx.from_numpy_matrix(adjacency_matrix)

	# Node degree
	node_degree = nx.degree(G)
	mean_degree = np.mean([node_degree[i] for i in node_degree])

	# Clustering coefficient 
	clustering_coefficient = nx.clustering(G)
	mean_clustcoeff = np.mean([clustering_coefficient[i] for i in clustering_coefficient])

	# Betweenness centrality
	betweenness_centrality = nx.betweenness_centrality(G)
	max_betweencent = np.mean([betweenness_centrality[i] for i in betweenness_centrality])

	# Maximum richclub coefficient
	richclub_coefficient = nx.rich_club_coefficient(G)
	max_richcoeff = np.max([richclub_coefficient[i] for i in richclub_coefficient])

	return mean_degree, mean_clustcoeff, max_betweencent, max_richcoeff

def get_graph_measures_pandas(in_files):
	import pandas as pd
	mean_degrees = list()
	mean_clustcoeffs = list()
	max_betweencents = list()
	max_richcoeffs = list()
	file_names = list()

	for in_file in in_files:
		mean_degree, mean_clustcoeff, max_betweencent, max_richcoeff = calculate_graph_measures(in_file)
		mean_degrees.append(mean_degree)
		mean_clustcoeffs.append(mean_clustcoeff)
		max_betweencents.append(max_betweencent)
		max_richcoeffs.append(max_richcoeff)
		file_names.append(in_file.split('/')[-1])
		
	df = pd.DataFrame(data={'mean_NodeDegree':mean_degrees,'mean_ClusteringCoefficient':mean_clustcoeffs,
		'max_BetweennessCentrality':max_betweencents,'max_RichClub':max_richcoeffs},index=file_names)

	return df 