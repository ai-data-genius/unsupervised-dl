import numpy as np
import tensorflow as tf


class kmeans:
    def __init__(self, n_clusters, max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def lloyd(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        #initialize the centroids by copying random points from the data
        centroids = np.array([X[i] for i in np.random.choice(n_samples, self.n_clusters, replace=False)], copy=True)

        print(centroids)

    def compression(self: 'Kmeans') -> None:
        pass

    def decrompression(self: 'Kmeans') -> None:
        pass

    def projection(self: 'Kmeans') -> None:
        pass

    def generation(self: 'Kmeans') -> None:
        pass


