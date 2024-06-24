import numpy as np
import matplotlib.pyplot as plt
from dataset.dataset_mnist import mnistData
from sklearn.datasets import fetch_openml
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit_transform(self, X):
        # Flatten each image in X
        X_flat = np.array([x.flatten() for x in X])

        # Center the data
        self.mean = np.mean(X_flat, axis=0)
        X_centered = X_flat - self.mean

        # Calculate covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvectors by descending eigenvalues
        sorted_index = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_index]
        sorted_eigenvectors = eigenvectors[:, sorted_index]

        # Select n_components eigenvectors
        self.components = sorted_eigenvectors[:, :self.n_components]

        # Transform the data
        X_reduced = np.dot(X_centered, self.components)

        return X_reduced

    def inverse_transform(self, X_reduced):
        # Inverser la transformation
        X_centered = np.dot(X_reduced, self.components.T)
        X_original = X_centered + self.mean
        return X_original

    def detect_clusters(self, X_reduced, threshold=0.5):
        # Détection des clusters basée sur les composants principaux
        distance_matrix = squareform(pdist(X_reduced, 'euclidean'))
        Z = linkage(distance_matrix, method='ward')
        clusters = fcluster(Z, t=threshold * np.max(Z[:, 2]), criterion='distance')
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
            cluster_indices = np.where(clusters == i+1)[0]
            sample_indices = np.random.choice(cluster_indices, 10, replace=False)
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
X = X_train.reshape(-1, 28, 28)
pca = PCA(n_components=2)

# Appliquer notre propre PCA
X_reduced = pca.fit_transform(X)

# Détecter les clusters
clusters = pca.detect_clusters(X_reduced)

# Visualiser les clusters
pca.visualize_clusters(X_reduced, clusters)

# Afficher des exemples d'images de chaque cluster
n_clusters = len(np.unique(clusters))
pca.plot_sample_images(X, clusters, n_clusters)




