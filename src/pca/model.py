import numpy as np
import matplotlib.pyplot as plt
from dataset.dataset_mnist import mnistData
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.datasets import fetch_openml
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

class PCAp:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit_transform(self, X):
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

    def inverse_transform(self, X_reduced):
        # Inverser la transformation
        X_centered = np.dot(X_reduced, self.components.T)
        X_original = X_centered + self.mean
        return X_original

    def detect_clusters(self, X_reduced, threshold=5.0):
        n_samples = X_reduced.shape[0]
        clusters = np.full(n_samples, -1, dtype=int)
        cluster_id = 0

        for i in range(n_samples):
            if clusters[i] == -1:
                clusters[i] = cluster_id
                for j in range(i + 1, n_samples):
                    if clusters[j] == -1:
                        distance = np.linalg.norm(X_reduced[i] - X_reduced[j])
                        if distance < threshold:
                            clusters[j] = cluster_id
                cluster_id += 1

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


    def compression(self: 'Kmeans') -> None:
        pass

    def decrompression(self: 'Kmeans') -> None:
        pass

    def projection(self: 'Kmeans') -> None:
        pass

    def generation(self: 'Kmeans') -> None:
        pass


# Utilisation de la classe PCA
X_train = mnistData().getTrainX()
X = X_train.reshape(X_train.shape[0], -1)  # (60000, 784)
# n_samples = 1000
# n_clusters = 3
#
# X = np.random.randn(n_samples, 2)
# X[:300] += 5
# X[300:600] += 10
# X[600:] += 15
pcap = PCAp(n_components=2)
pca = PCA(n_components=2)

# Appliquer notre propre PCA
X_reduced = pcap.fit_transform(X)

# Détecter les clusters
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_reduced)
clusters = kmeans.predict(X_reduced)

# Visualiser les clusters
pcap.visualize_clusters(X_reduced, clusters)

# Afficher des exemples d'images de chaque cluster
# n_clusters = 10
# pca.plot_sample_images(X, clusters, n_clusters)




