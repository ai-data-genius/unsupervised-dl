import numpy as np
import matplotlib.pyplot as plt
from dataset.dataset_mnist import mnistData
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def compress(self, X):
        # Centrer les données
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Assurer que X_centered est en 2D
        if X_centered.ndim != 2:
            raise ValueError("Les données doivent être en 2D")

        # Calculer la matrice de covariance
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Calculer les valeurs propres et les vecteurs propres
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Trier les vecteurs propres par ordre décroissant des valeurs propres
        sorted_index = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_index]
        sorted_eigenvectors = eigenvectors[:, sorted_index]

        # Sélectionner les n_components vecteurs propres
        self.components = sorted_eigenvectors[:, :self.n_components]

        # Transformer les données
        X_reduced = np.dot(X_centered, self.components)

        return X_reduced

    def decompress(self, X_reduced):
        # Inverser la transformation
        X_original = np.dot(X_reduced, self.components.T) + self.mean
        return X_original

    def cluster_with_pca(self, X_reduced, n_clusters):
        spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
        clusters = spectral_clustering.fit_predict(X_reduced)
        return clusters
    def visualize_clusters(self, X_reduced, clusters):
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Clusters detected by PCA')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()

    def plot_sample_images(self, X, clusters, n_clusters):
        plt.figure(figsize=(12, 8))
        for i in range(n_clusters):
            cluster_indices = np.where(clusters == i + 1)[0]
            if len(cluster_indices) == 0:
                continue
            sample_size = min(len(cluster_indices), 10)
            sample_indices = np.random.choice(cluster_indices, sample_size, replace=False)
            for j, index in enumerate(sample_indices):
                plt.subplot(n_clusters, 10, i * 10 + j + 1)
                plt.imshow(X[index].reshape(28, 28), cmap='gray')
                plt.axis('off')
        plt.suptitle('Sample images from each cluster')
        plt.show()

    def determine_optimal_components(X, variance_threshold=0.95):
        mean = np.mean(X, axis=0)
        X_centered = X - mean

        # covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # eigenvalues in descending order
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]

        # cumulative explained variance
        cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)

        # number of components needed to reach the variance threshold
        optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1

        return optimal_components


# # Utilisation de la classe PCA
# X_train = mnistData().getTrainX()
# X = X_train[:20000].reshape(X_train[:20000].shape[0], -1)  # (60000, 784)
#
# optimal_components = PCA.determine_optimal_components(X, variance_threshold=0.95)
# print(f"Optimal number of components to retain 95% variance: {optimal_components}")
# pcap = PCA(n_components=optimal_components)
#
# # PCA
# X_reduced = pcap.fit_transform(X)
# X = pcap.inverse_transform(X_reduced)
#
# #Cluster
# clusters = pcap.cluster_with_pca(X_reduced, 10)
#
# # Visualiser les clusters
# pcap.visualize_clusters(X_reduced, clusters)
#
# # Afficher des exemples d'images de chaque cluster
# n_clusters = 10
# pcap.plot_sample_images(X, clusters, n_clusters)




