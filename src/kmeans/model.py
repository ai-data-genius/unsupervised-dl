import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class kmeans:
    def __init__(self, n_clusters, max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.membership = None

    def lloyd(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        #initialize the centroids by copying random points from the data
        centroids = np.array([X[i] for i in np.random.choice(n_samples, self.n_clusters, replace=False)], copy=True)

        print(centroids)

        #initialize the membership vector
        membership = np.zeros(n_samples)

        for i in range(self.max_iter):
            #assign the points to the closest cluster centroid by computing the Euclidean distance
            for j in range(n_samples):
                membership[j] = np.argmin([np.linalg.norm(X[j] - c) for c in centroids])

            #update the centroids
            for j in range(self.n_clusters):
                centroids[j] = np.mean(X[membership == j], axis=0)

        self.centroids = centroids
        self.membership = membership


        return centroids

    def plot_clusters(self, X):
        plt.scatter(X[:, 0], X[:, 1], c=self.membership)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x')
        plt.show()




    def compression(self: 'Kmeans') -> None:
        pass

    def decrompression(self: 'Kmeans') -> None:
        pass

    def projection(self: 'Kmeans') -> None:
        pass

    def generation(self: 'Kmeans') -> None:
        pass


