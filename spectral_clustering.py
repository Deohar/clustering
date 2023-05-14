import numpy as np
from numpy.linalg import norm


class SpectralClustering:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def fit_predict(self, affinity_matrix):
        laplacian_matrix = self._compute_unnormalized_laplacian(affinity_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
        sorted_indices = np.argsort(eigenvalues)[:self.num_clusters]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        centroids = self._kmeans_clustering(sorted_eigenvectors)

        cluster_labels = self._assign_clusters(sorted_eigenvectors, centroids)

        return cluster_labels

    def _compute_unnormalized_laplacian(self, affinity_matrix):
        degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
        laplacian_matrix = degree_matrix - affinity_matrix
        return laplacian_matrix

    def _kmeans_clustering(self, data):
        centroids, _ = kmeans(data, self.num_clusters)
        return centroids

    def _assign_clusters(self, data, centroids):
        cluster_labels = np.argmin(norm(data[:, np.newaxis] - centroids, axis=-1), axis=-1)
        return cluster_labels