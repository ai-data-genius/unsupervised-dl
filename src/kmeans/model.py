import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans

class KMeans:
    def __init__(self, n_clusters, max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.membership = None
        self.centroids = None

    def lloyd(self, X, tol=1e-4):
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)

        X_flat = X_flat / 255.0

        centroids = X_flat[np.random.choice(n_samples, self.n_clusters, replace=False)]
        centroids_old = np.zeros_like(centroids)


        for iteration in tqdm(range(self.max_iter), desc="Lloyd's Algorithm Progress"):

            if np.linalg.norm(centroids - centroids_old) <= tol:
                break

            centroids_old = centroids.copy()

            # for i in range(n_samples):
            #     membership[i] = np.argmin(np.linalg.norm(X_flat[i] - centroids, axis=1))
            membership = np.array(
                list(map(lambda i: np.argmin(np.linalg.norm(X_flat[i] - centroids, axis=1)), range(n_samples))))

            #map instead of for loop from func_tools instead of for loop


            # for i in range(self.n_clusters):
            #     centroids[i] = np.mean(X_flat[membership == i], axis=0)

            #map instead of for loop from func_tools instead of for loop
            centroids = np.array(list(map(lambda i: np.mean(X_flat[membership == i], axis=0), range(self.n_clusters))))


        self.centroids = centroids
        self.membership = membership

    def plot_clusters(self, X, Y):
        #plot the a histogram the number or percentage of each class in each cluster
        n_samples = X.shape[0]


        n_classes = len(np.unique(Y))
        n_clusters = self.n_clusters
        membership = self.membership

        #initialize the histogram
        hist = np.zeros((n_clusters, n_classes))

        for i in range(n_samples):
            hist[int(membership[i]), int(Y[i])] += 1

        #normalize the histogram
        hist = hist / np.sum(hist, axis=1)[:, None]

        #plot the histogram
        fig, ax = plt.subplots(1, n_clusters, figsize=(10, 2))
        for i in range(n_clusters):
            ax[i].bar(np.arange(n_classes), hist[i])
            ax[i].set_title('Cluster %d' % i)
            ax[i].set_xticks(np.arange(n_classes))
            ax[i].set_xticklabels(np.arange(n_classes))

        plt.show()

        #plot the centroids


        fig, ax = plt.subplots(1, self.n_clusters, figsize=(10, 2))
        for i in range(self.n_clusters):
            ax[i].imshow(self.centroids[i].reshape(28, 28), cmap='gray')
            ax[i].axis('off')
        plt.show()

    def compress(self, image):
        image_flat = image.reshape(-1)
        return np.argmin(np.linalg.norm(image_flat - self.centroids, axis=1))

    def decompress(self, id_cluster):
        if self.centroids is None:
            raise ValueError("Centroids are not initialized. Run the lloyd method first.")

        if id_cluster >= self.n_clusters:
            raise ValueError(f"Invalid id_cluster. It should be between 0 and {self.n_clusters - 1}.")

        centroid = self.centroids[id_cluster]
        decompressed_image =  centroid.reshape(28, 28)

        plt.imshow(decompressed_image, cmap='gray')
        plt.axis('off')
        plt.show()


    def generate(self, steps=10):
        if self.centroids is None:
            raise ValueError("Centroids are not initialized. Run the lloyd method first.")

        images = []
        #random value between 0 and n_clusters int
        random_value1 = np.random.randint(0, self.n_clusters)
        #random_value2 should be different from random_value1, so ensure it is different
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

# Example usage:
# kmeans = KMeans(n_clusters=10, max_iter=1000)
# kmeans.lloyd(data)  # where data is your dataset
# kmeans.plot_clusters(reduced_data)  # where reduced_data is your PCA-reduced dataset
