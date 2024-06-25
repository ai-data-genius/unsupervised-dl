from kmeans.model import KMeans
from dataset.dataset_mnist import mnistData
from dataset.dataset_toy import toyData
from pca.model import PCA
import matplotlib.pyplot as plt
import numpy as np

def main():
    mnist_dataset = mnistData()
    toy_dataset = toyData()
    X_toy = toy_dataset.getX()
    X = mnist_dataset.getTrainX()
    Y = mnist_dataset.getTrainY()


    model_choice = input("Enter the model to run ('kmeans' or 'pca'): ").strip().lower()

    if model_choice == 'kmeans':
        # initialize the model
        kmeans = KMeans(20, max_iter=30)
        kmeans.lloyd(X)

        # plot the clusters
        kmeans.plot_clusters(X, Y)

        # plot first image of X
        plt.imshow(X[1])
        plt.show()

        compressed = kmeans.compress(X[1])
        # decompress the image
        kmeans.decompress(compressed)

        # generate a new image
        X_gen = kmeans.generate(steps=25)

    elif model_choice == 'pca':
        X = X[:5000].reshape(X[:5000].shape[0], -1)  # (60000, 784)
        # Determine the optimal number of components
        optimal_components = PCA.determine_optimal_components(X, variance_threshold=0.95)
        print(f"Optimal number of components to retain 95% variance: {optimal_components}")

        # Use the optimal number of components for PCA
        pcap = PCA(n_components=optimal_components)

        # Apply PCA and compress data
        X_reduced = pcap.compress(X)

        # Decompress data
        X_reconstructed = pcap.decompress(X_reduced)

        # Detect clusters on reduced data
        clusters = pcap.cluster_with_pca(X_reduced, 10)

        # Visualize clusters
        pcap.visualize_clusters(X_reduced, clusters)

        # Display sample images from each cluster
        pcap.plot_sample_images(X_reconstructed, clusters, n_clusters=10)

    else:
        print("Invalid model choice. Please enter 'kmeans' or 'pca'.")

if __name__ == "__main__":
    main()

