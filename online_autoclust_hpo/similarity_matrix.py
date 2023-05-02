# from: https://github.com/jaumpedro214/posts/blob/main/ensamble_clustering/simlarity_matrix.py

from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
import numpy as np


class ClusterSimilarityMatrix():
    """
    A class to calculate and store a similarity matrix for cluster assignments.
    """
    
    def __init__(self) -> None:
        """
        Initialize a ClusterSimilarityMatrix instance with an _is_fitted attribute.
        """
        self._is_fitted = False

    def fit(self, y_clusters):
        """
        Fit the similarity matrix to the provided cluster assignments. If the matrix
        is not fitted yet, initializes the similarity matrix; otherwise, updates
        the existing matrix with new cluster assignments.

        Parameters
        ----------
        y_clusters : array-like
            An array of cluster assignments for a set of data points.

        Returns
        -------
        self : ClusterSimilarityMatrix
            The updated ClusterSimilarityMatrix instance.
        """
        if not self._is_fitted:
            self._is_fitted = True
            self.similarity = self.to_binary_matrix(y_clusters)
            return self

        self.similarity += self.to_binary_matrix(y_clusters)

    def to_binary_matrix(self, y_clusters):
        """
        Convert the provided cluster assignments to a binary similarity matrix.

        Parameters
        ----------
        y_clusters : array-like
            An array of cluster assignments for a set of data points.

        Returns
        -------
        binary_matrix : np.ndarray
            A binary similarity matrix representing the cluster assignments.
        """
        y_reshaped = np.expand_dims(y_clusters, axis=-1)
        return (cdist(y_reshaped, y_reshaped, 'cityblock')==0).astype(int)

def clustering_ensemble(lists_of_labels, MIN_PROBABILITY = 0.6):
    """
    Ensembles the clustering results by computing a similarity matrix and
    applying a threshold to the normalized similarity matrix to create a
    graph, then using connected components to obtain the final cluster
    assignments.

    Args:
    lists_of_labels : list of lists
        A list of lists of predicted labels from every model.
    MIN_PROBABILITY : float, optional (default=0.6)
        The probability of co-occurrence used as a threshold for graph
        creation.

    Returns:
    y_ensemble : np.ndarray
        An array containing the ensembled cluster assignments. The last
        element of the array is the clustering prediction from the ensemble.

    References:
    [1] Pedro, J. (2022). How to ensemble Clustering Algorithms. Accessed on
        01.04.2023 from
        https://towardsdatascience.com/how-to-ensemble-clustering-algorithms-bf78d7602265.
    [2] Pedro, J. (2022). Ensemble clustering. A Github repostiory accessed on
        01.04.2023 from
        https://github.com/jaumpedro214/posts/blob/main/ensamble_clustering/
    """
    clt_sim_matrix = ClusterSimilarityMatrix()
    
    for clustering_result in lists_of_labels:
        clt_sim_matrix.fit(clustering_result) 
    
    MIN_PROBABILITY = MIN_PROBABILITY

    sim_matrix = clt_sim_matrix.similarity
    norm_sim_matrix = sim_matrix/sim_matrix.diagonal()
    graph = (norm_sim_matrix>MIN_PROBABILITY).astype(int)

    _, y_ensemble = connected_components(graph, directed=False, return_labels=True )
    return y_ensemble[-1]
