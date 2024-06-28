import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans


class KMeans:
    def __init__(self: 'KMeans', n_clusters, _type="image", max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.membership = None
        self.centroids = None
        self._type = _type

    def lloyd(self: "KMeans", X, tol=1e-4):
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)

        if self._type == "image":
            print("Image")
            X_flat = X_flat / 255.0
        centroids = X_flat[np.random.choice(n_samples, self.n_clusters, replace=False)]
        centroids_old = np.zeros_like(centroids)

        for iteration in tqdm(range(self.max_iter), desc="Lloyd's Algorithm Progress"):
            centroids_old = centroids.copy()
            membership = np.array(
                list(map(lambda i: np.argmin(np.linalg.norm(X_flat[i] - centroids, axis=1)), range(n_samples))))

            for i in range(self.n_clusters):
                if np.any(membership == i):
                    centroids[i] = np.mean(X_flat[membership == i], axis=0)
                else:
                    # Reinitialize empty cluster
                    centroids[i] = X_flat[np.random.choice(n_samples)]

        self.centroids = centroids
        self.membership = membership

    def projection(self: "KMeans", **kwargs):
        (self.projection_1d, self.projection_2d)[self._type == "image"](**kwargs)

    def projection_1d(self: "KMeans", X):
        plt.scatter(X[:, 0], X[:, 1], c=self.membership)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x')
        plt.show()

    def projection_2d(self: "KMeans", X, Y):
        # Plot a histogram of the number or percentage of each class in each cluster
        n_samples = X.shape[0]
        n_classes = len(np.unique(Y))
        n_clusters = self.n_clusters
        membership = self.membership

        # Initialize the histogram
        hist = np.zeros((n_clusters, n_classes))

        for i in range(n_samples):
            hist[int(membership[i]), int(Y[i])] += 1

        # Normalize the histogram
        hist = hist / np.sum(hist, axis=1)[:, None]

        # Plot the histogram
        for i in range(0, self.n_clusters, 10):
            fig, ax = plt.subplots(1, 10, figsize=(20, 5))
            for j in range(10):
                if i + j < n_clusters:  # Ensure we don't go out of bounds
                    ax[j].bar(np.arange(n_classes), hist[i + j])
                    ax[j].set_ylim(0, 1)  # Set y-axis limits
                    ax[j].set_title('Cluster %d' % (i + j))
                    ax[j].set_xticks(np.arange(n_classes))
                    ax[j].set_xticklabels(np.arange(n_classes))
                else:
                    ax[j].axis('off')  # Turn off the axis if there is no cluster to display
            plt.tight_layout()
            plt.show()

        # Plot the centroids
        for i in range(0, self.n_clusters, 10):
            fig, ax = plt.subplots(1, 10, figsize=(20, 5))
            for j in range(10):
                if i + j < self.n_clusters:  # Ensure we don't go out of bounds
                    ax[j].imshow(self.centroids[i + j].reshape(32, 32, 3), cmap='gray')
                    ax[j].axis('off')
                    ax[j].set_title('Cluster %d' % (i + j))
                else:
                    ax[j].axis('off')  # Turn off the axis if there is no cluster to display
            plt.tight_layout()
            plt.show()

    def compress(self: "KMeans", image):
        image_flat = image.reshape(-1)
        image_flat = image_flat / 255.0

        id_cluster = np.argmin(np.linalg.norm(image_flat - self.centroids, axis=1))
        return id_cluster

    def decompress(self: "KMeans", id_cluster: int):
        if self.centroids is None:
            raise ValueError("Centroids are not initialized. Run the lloyd method first.")

        centroid = self.centroids[id_cluster]
        decompressed_image = centroid.reshape(32, 32, 3)
        return decompressed_image

    def generate(self: "KMeans", steps: int = 10):
        if self.centroids is None:
            raise ValueError("Centroids are not initialized. Run the lloyd method first.")

        images = []
        random_value1 = np.random.randint(0, self.n_clusters)
        while True:
            random_value2 = np.random.randint(0, self.n_clusters)
            if random_value2 != random_value1:
                break

        for alpha in np.linspace(0, 1, steps):
            interpolated_image = (1 - alpha) * self.centroids[random_value1] + alpha * self.centroids[random_value2]
            images.append(interpolated_image.reshape(28, 28))

        fig, ax = plt.subplots(1, steps, figsize=(20, 2))
        for i in range(steps):
            ax[i].imshow(images[i], cmap='gray')
            ax[i].axis('off')
        plt.show()

    def save_weights(self):
        np.save(f'kmeans/models/model_{self.n_clusters}_{self.max_iter}', self.centroids)

    def load_weights(self, file_path):
        self.centroids = np.load(file_path)
