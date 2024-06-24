import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

class kmeans:
    def __init__(self, n_clusters, max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.membership = None



    def lloyd(self, X, tol=1e-6):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # initialize the centroids by copying random points from the data
        centroids = np.array([X[i] for i in np.random.choice(n_samples, self.n_clusters, replace=False)], copy=True)
        centroids_old = np.zeros((self.n_clusters, n_features))


        # initialize the membership vector
        membership = np.zeros(n_samples)

        # tol = 1e-4 Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
        while np.linalg.norm(centroids - centroids_old) > tol:
            centroids_old = centroids.copy()


            # assign each point to the closest centroid
            for i in tqdm(range(n_samples)):
                membership[i] = np.argmin(np.linalg.norm(X[i] - centroids, axis=1))

            # update the centroids
            for i in tqdm(range(self.n_clusters)):
                centroids[i] = np.mean(X[membership == i], axis=0)

        self.centroids = centroids
        self.membership = membership

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


