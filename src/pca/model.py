import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.cluster_centers_ = None

    def compress(self, X):
        # Centrer les données
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Assurer que X_centered est en 2D
        if X_centered.ndim != 2:
            raise ValueError("Les données doivent être en 2D")

        # Calculer la matrice de covariance
        n_samples = X_centered.shape[0]
        covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

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

    def cluster_with_pca(self, X_reduced, Y):
        # Utilisation des étiquettes pour guider le clustering
        n_clusters = len(np.unique(Y))
        cluster_centers = np.zeros((n_clusters, X_reduced.shape[1]))

        # Calculer les centres de clusters basés sur les moyennes des chiffres
        for digit in range(n_clusters):
            digit_indices = np.where(Y == digit)[0]
            cluster_centers[digit] = np.mean(X_reduced[digit_indices], axis=0)

        self.cluster_centers_ = cluster_centers

        # Assigner chaque point à son cluster correspondant
        clusters = np.argmin(np.linalg.norm(X_reduced[:, np.newaxis] - self.cluster_centers_, axis=2), axis=1)

        return clusters

    def cluster_with_pca_toy(self, X_reduced, n_clusters, max_iter=100):
        # Initialiser les centres de cluster aléatoirement parmi les points
        np.random.seed(42)
        initial_indices = np.random.choice(X_reduced.shape[0], n_clusters, replace=False)
        cluster_centers = X_reduced[initial_indices]
        clusters = np.zeros(X_reduced.shape[0])

        for _ in tqdm(range(max_iter), desc="Running Manual Clustering"):
            # Assigner chaque point au cluster le plus proche
            distances = np.sqrt(((X_reduced[:, np.newaxis] - cluster_centers) ** 2).sum(axis=2))
            clusters = np.argmin(distances, axis=1)

            # Mettre à jour les centres de cluster
            for i in range(n_clusters):
                points_in_cluster = X_reduced[clusters == i]
                if len(points_in_cluster) > 0:
                    cluster_centers[i] = points_in_cluster.mean(axis=0)

        self.cluster_centers_ = cluster_centers
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
        n_samples = X_centered.shape[0]
        covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # eigenvalues in descending order
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]

        # cumulative explained variance
        cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)

        # number of components needed to reach the variance threshold
        optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1

        return optimal_components

    def generate_image(self, n_images=1):
        # Vérifier que fit_transform a été appelé
        if self.components is None:
            raise ValueError("Le modèle doit être ajusté avant de générer des images.")

        # Générer des points aléatoires autour des centres des clusters
        if self.cluster_centers_ is None:
            raise ValueError("Les centres des clusters doivent être définis avant de générer des images.")

        images = []
        for _ in range(n_images):
            cluster_idx = np.random.choice(len(self.cluster_centers_))
            cluster_center = self.cluster_centers_[cluster_idx]
            X_reduced_random = cluster_center + 0.1 * np.random.randn(self.n_components)
            X_generated = self.decompress(X_reduced_random.reshape(1, -1))
            images.append(X_generated)

        return np.array(images).reshape(-1, 28, 28)

    def find_cluster_centers(self, X_reduced, n_clusters):
        # Initialisation structurée des clusters
        initial_indices = np.random.choice(X_reduced.shape[0], n_clusters, replace=False)
        self.cluster_centers_ = X_reduced[initial_indices]

        for _ in range(10):  # Nombre d'itérations pour ajuster les centres
            clusters = np.argmin(np.linalg.norm(X_reduced[:, np.newaxis] - self.cluster_centers_, axis=2), axis=1)
            new_centers = np.array([X_reduced[clusters == i].mean(axis=0) for i in range(n_clusters)])
            if np.allclose(self.cluster_centers_, new_centers):
                break
            self.cluster_centers_ = new_centers

    def latent_space_walk(self, n_steps=10, n_dimensions=2):
        if self.components is None or self.mean is None:
            raise ValueError("PCA must be fitted before performing latent space walk.")

        # Create a grid in the latent space
        linspace = np.linspace(-3, 3, n_steps)
        grid = np.meshgrid(*[linspace for _ in range(n_dimensions)])
        grid_flat = np.column_stack([g.ravel() for g in grid])

        # Pad the grid points if necessary
        if n_dimensions < self.n_components:
            grid_flat = np.pad(grid_flat, ((0, 0), (0, self.n_components - n_dimensions)))

        # Generate images from the grid points
        generated_images = self.decompress(grid_flat)

        # Plot the generated images
        fig, axes = plt.subplots(n_steps, n_steps, figsize=(15, 15))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(generated_images[i].reshape(28, 28), cmap='viridis')
            ax.axis('on')

        plt.suptitle('Latent Space Walk')
        plt.tight_layout()
        plt.show()
