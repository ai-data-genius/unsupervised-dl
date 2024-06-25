import matplotlib.pyplot as plt
import numpy as np
import sys

from autoencoder.model import AE
from dataset.dataset_mnist import mnistData
from dataset.dataset_toy import toyData
from kmeans.model import KMeans
from pca.model import PCA


if __name__ == '__main__':
    mnist_dataset = mnistData()
    toy_dataset = toyData()

    X_toy = toy_dataset.getX()
    X_train = mnist_dataset.getTrainX()
    Y_train = mnist_dataset.getTrainY()
    X_test = mnist_dataset.getTestX()
    Y_test = mnist_dataset.getTestY()

    model_choice = input("Enter the model to run ('kmeans', 'pca', 'autoencoder'): ").strip().lower()


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
        # compressed = kmeans.compress(X_train[1])
        # # decompress the image
        # kmeans.decompress(compressed)
        #
        # # generate a new image
        # X_gen = kmeans.generate(steps=25)

    elif model_choice == "pca":
        X_pca = X_train.reshape(X_train.shape[0], -1)  # (60000, 784)
        # Determine the optimal number of components
        optimal_components = PCA.determine_optimal_components(X_pca, variance_threshold=0.95)
        print(f"Optimal number of components to retain 95% variance: {optimal_components}")

        # Use the optimal number of components for PCA
        pcap = PCA(n_components=optimal_components)

        # Apply PCA and compress data
        X_reduced = pcap.compress(X_pca)

        # Decompress data
        X_reconstructed = pcap.decompress(X_reduced)

        # Affichage des images avant et après compression
        fig, axes = plt.subplots(2, 10, figsize=(15, 4))
        fig.suptitle('Images avant et après compression/décompression')

        for i in range(10):
            axes[0, i].imshow(X_pca[i].reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')

            axes[1, i].imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')

        plt.show()

        # Detect clusters on reduced data
        clusters = pcap.cluster_with_pca(X_reduced, Y_train)

        # Visualize clusters
        pcap.visualize_clusters(X_reduced, clusters)

        # Display sample images from each cluster
        pcap.plot_sample_images(X_reconstructed, clusters, n_clusters=10)
        # Générer une nouvelle image
        pcap.find_cluster_centers(X_reduced, n_clusters=10)
        X_generated = pcap.generate_image(n_images=5)
        plt.imshow(X_generated[0].reshape(28, 28), cmap='gray')
        plt.title('Generated Image')
        plt.show()

    elif model_choice == 'autoencoder':
        mnist_dataset = mnistData()
        #get only first 1000 samples
        X_train = mnist_dataset.getTrainX()
        Y_train = mnist_dataset.getTrainY()
        X_test = mnist_dataset.getTestX()
        Y_test = mnist_dataset.getTestY()
        ae = AE(32, (784,), 784)


        ae.build()
        X_train, X_test = ae.standardize(X_train, X_test)
        ae.fit(X_train, X_test, 100, 32)

        ae.projection(X_test, ae.autoencoder.predict(X_test), Y_test, graph=False, n=20)
        ae.projection(X_test, ae.autoencoder.predict(X_test), Y_test, graph=True)

        ae.generation(20)
    else:
        print("Invalid model choice. Please enter 'kmeans' or 'pca' or 'autoencoder'.")

