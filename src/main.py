import matplotlib.pyplot as plt
import numpy as np
import sys

from autoencoder.model import AE
from dataset.dataset_mnist import mnistData
from dataset.dataset_toy import toyData
from kmeans.model import KMeans
from pca.model import PCA


if __name__ == "__main__":
    mnist_dataset = mnistData()
    toy_dataset = toyData()

    X_toy = toy_dataset.getX()
    X_train = mnist_dataset.getTrainX()
    Y_train = mnist_dataset.getTrainY()
    X_test = mnist_dataset.getTestX()
    Y_test = mnist_dataset.getTestY()

    model_choice = input("Enter the model to run ('kmeans' or 'pca'): ").strip().lower()

    if model_choice == "kmeans":
        dataset_choice = input("Enter the dataset to run ('mnist' or 'toy'): ").strip().lower()

        if dataset_choice == "mnist":
            kmeans = KMeans(20, max_iter=100)
            kmeans.lloyd(X_train)
            kmeans.projection(X=X_train, Y=Y_train)

            compressed = kmeans.compress(X_train[1])
            kmeans.decompress(compressed)
            kmeans.generate(steps=25)
        elif dataset_choice == "toy":
            plt.scatter(X_toy[:, 0], X_toy[:, 1])
            plt.show()

            kmeans = KMeans(3, "point")
            kmeans.lloyd(X_toy)
            kmeans.projection(X=X_toy)
    elif model_choice == "pca":
        X_pca = X_train[:5000].reshape(X_train[:5000].shape[0], -1)  # (60000, 784)
        # Determine the optimal number of components
        optimal_components = PCA.determine_optimal_components(X_pca, variance_threshold=0.95)
        print(f"Optimal number of components to retain 95% variance: {optimal_components}")

        # Use the optimal number of components for PCA
        pcap = PCA(n_components=optimal_components)

        # Apply PCA and compress data
        X_reduced = pcap.compress(X_pca)

        # Decompress data
        X_reconstructed = pcap.decompress(X_reduced)

        # Detect clusters on reduced data
        clusters = pcap.cluster_with_pca(X_reduced, 10)

        # Visualize clusters
        pcap.visualize_clusters(X_reduced, clusters)

        # Display sample images from each cluster
        pcap.plot_sample_images(X_reconstructed, clusters, n_clusters=10)
    elif model_choice == "autoencoder":
        ae = AE(32, (784,), 784)
        ae.build()
        X_train, X_test = ae.standardize(X_train, X_test)
        ae.fit(X_train, X_test, 20, 1028)
        ae.projection(X_test, ae.autoencoder.predict(X_test))
    else:
        print("Invalid model choice. Please enter 'kmeans' or 'pca'.")
