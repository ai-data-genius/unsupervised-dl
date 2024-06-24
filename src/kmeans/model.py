import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class KMeans:
    def __init__(self, n_clusters, max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.membership = None
        self.centroids = None

    def lloyd(self, X, tol=1e-4):
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)

        # Initialize the centroids by copying random points from the data
        centroids = X_flat[np.random.choice(n_samples, self.n_clusters, replace=False)]
        centroids_old = np.zeros_like(centroids)

        # Initialize the membership vector
        membership = np.zeros(n_samples)

        # Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
        for iteration in tqdm(range(self.max_iter), desc="Lloyd's Algorithm Progress"):
            if np.linalg.norm(centroids - centroids_old) <= tol:
                break

            centroids_old = centroids.copy()

            # Assign each point to the closest centroid
            for i in range(n_samples):
                membership[i] = np.argmin(np.linalg.norm(X_flat[i] - centroids, axis=1))

            # Update the centroids
            for i in range(self.n_clusters):
                points_in_cluster = X_flat[membership == i]
                if len(points_in_cluster) > 0:
                    centroids[i] = np.mean(points_in_cluster, axis=0)
                else:  # If a cluster gets no points, reinitialize its centroid
                    centroids[i] = X_flat[np.random.choice(n_samples)]

        self.centroids = centroids
        self.membership = membership

    def plot_clusters(self, reduced_data):
        reduced_data = reduced_data.reshape(reduced_data.shape[0], -1)

        plt.figure(figsize=(10, 8))
        for i in range(self.n_clusters):
            cluster_data = reduced_data[self.membership == i]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i}')
        plt.legend()
        plt.title('Clusters K-means sur MNIST (réduction de dimension avec PCA)')
        plt.xlabel('Composante principale 1')
        plt.ylabel('Composante principale 2')
        plt.show()

# Example usage:
# kmeans = KMeans(n_clusters=10, max_iter=1000)
# kmeans.lloyd(data)  # where data is your dataset
# kmeans.plot_clusters(reduced_data)  # where reduced_data is your PCA-reduced dataset
