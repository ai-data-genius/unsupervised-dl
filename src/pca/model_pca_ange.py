import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import mnist
import os

class PCA:
    def __init__(self, n_components, cov_matrix_path='cov_matrix.npy', means_path='means.npy'):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.means_ = None
        self.cov_matrix_path = cov_matrix_path
        self.means_path = means_path

    def fit(self, X):
        # Calculer ou charger la matrice de covariance et les moyennes
        if os.path.exists(self.cov_matrix_path) and os.path.exists(self.means_path):
            covariance_matrix = np.load(self.cov_matrix_path)
            self.means_ = np.load(self.means_path)
            print(f"Loaded covariance matrix from {self.cov_matrix_path} and means from {self.means_path}")
        else:
            covariance_matrix = self.manual_covar(X)
            np.save(self.cov_matrix_path, covariance_matrix)
            np.save(self.means_path, self.means_)
            print(f"Saved covariance matrix to {self.cov_matrix_path} and means to {self.means_path}")

        # Calculer les valeurs et vecteurs propres
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

        # Trier les valeurs propres en ordre décroissant
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalues = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]

        # Sélectionner les n_components premiers vecteurs propres
        self.components_ = sorted_eigenvectors[:, :self.n_components]
        self.explained_variance_ = sorted_eigenvalues[:self.n_components]

    def manual_covar(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Calculer les moyennes de chaque dimension
        self.means_ = np.mean(X, axis=0)

        # Centrer les données
        centered_data = X - self.means_

        # Initialiser la matrice de covariance
        covariance_matrix = np.zeros((n_features, n_features))

        # Calculer la matrice de covariance avec tqdm
        for i in range(n_features):
            for j in range(n_features):
                covariance_matrix[i, j] = np.sum(centered_data[:, i] * centered_data[:, j]) / (n_samples - 1)

        return covariance_matrix

    def transform(self, X):
        # Centrer les données en utilisant les moyennes calculées pendant l'ajustement
        centered_data = X - self.means_
        # Projeter les données sur les composants principaux
        return np.dot(centered_data, self.components_)

    def compression(self, X):
        # Utiliser les valeurs propres pour la compression
        centered_data = X - self.means_
        compressed_data = np.dot(centered_data, self.components_)
        return compressed_data

    def decompression(self, compressed_data):
        # Utiliser les vecteurs propres pour la décompression
        decompressed_data = np.dot(compressed_data, self.components_.T) + self.means_
        return decompressed_data

    def generate_new_images(self, latent_points):
        # Décompresser ces points pour obtenir de nouvelles images
        generated_images = self.decompression(latent_points)
        return generated_images

    def visualize_latent_space(self, X, y):
        # Transformer les données pour les projeter dans l'espace des composantes principales
        compressed_data = self.transform(X)

        # Utiliser les deux premières composantes principales pour la visualisation
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=compressed_data[:, 0], y=compressed_data[:, 1], hue=y, palette=sns.color_palette("hsv", 10), legend='full', alpha=0.6)
        plt.xlabel('Composante principale 1')
        plt.ylabel('Composante principale 2')
        plt.title('Visualisation de l\'espace latent avec les deux premières composantes principales')
        plt.show()

        return compressed_data

# Charger le dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normaliser les données
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Aplatir les images (passer de 28x28 à 784)
X_train_flat = X_train.reshape(-1, 28*28)
X_test_flat = X_test.reshape(-1, 28*28)

# Appliquer la PCA
n_components = 154  # Par exemple, réduire à 64 dimensions
pca = PCA(n_components=n_components)
pca.fit(X_train_flat)
compressed_data = pca.compression(X_train_flat)
decompressed_data = pca.decompression(compressed_data)

# Visualiser l'espace latent
compressed_data = pca.visualize_latent_space(X_train_flat, y_train)

# Filtrer les points latents correspondant aux chiffres "4"
latent_4s = compressed_data[y_train == 0]

# Générer de nouveaux points autour des points latents de "4"
n_new_images = 10
new_latent_points = np.random.normal(loc=np.mean(latent_4s, axis=0), scale=np.std(latent_4s, axis=0), size=(n_new_images, n_components))

# Générer de nouvelles images de "4"
new_images_4 = pca.generate_new_images(new_latent_points)

# Afficher les nouvelles images générées de "4"
fig, axes = plt.subplots(1, n_new_images, figsize=(20, 3))
for i, ax in enumerate(axes):
    ax.imshow(new_images_4[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Generated 0")
    ax.axis('off')
plt.show()

