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