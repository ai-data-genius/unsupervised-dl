import matplotlib.pyplot as plt
import numpy as np
import sys

from autoencoder.mnist.model import AE as AEMnist
from autoencoder.pokemon.model_pokemon import AE as AEPokemon
from variable_autoencoder.mnist.model import VAE as VAEMnist
from variable_autoencoder.pokemon.model import VAE as VAEPokemon
from dataset.dataset_mnist import mnistData
from dataset.dataset_toy import toyData
from kmeans.model import KMeans
from pca.model import PCA
from Self_Organizing_Maps.model import SOM
from math import sqrt
from src.dataset.dataset_pokemon import PokemonData
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    mnist_dataset = mnistData()
    toy_dataset = toyData()

    X_toy = toy_dataset.getX()
    X_train = mnist_dataset.getTrainX()
    Y_train = mnist_dataset.getTrainY()
    X_test = mnist_dataset.getTestX()
    Y_test = mnist_dataset.getTestY()

    pd = PokemonData(image_size=(32,32))
    X_train_pokemon = pd.getTrainX()
    X_test_pokemon = pd.getTestX()
    Y_train_pokemon = pd.getTrainY()
    Y_test_pokemon = pd.getTestY()

    model_choice = input("Enter the model to run ('kmeans', 'pca', 'autoencoder', 'vae', 'som'): ").strip().lower()

    if model_choice == "kmeans":
        dataset_choice = input("Enter the dataset to run ('mnist' or 'toy' or 'pokemon'): ").strip().lower()

        if dataset_choice == "mnist":
            kmeans = KMeans(100, max_iter=20)
            kmeans.lloyd(X_train)
            kmeans.projection(X=X_train, Y=Y_train)

            compressed = kmeans.compress(X_train[1])
            kmeans.decompress(compressed)
            kmeans.generate(steps=25)
        if dataset_choice == "pokemon":
            pokemon_dataset = PokemonData(image_size=(32, 32))

            X_train = pokemon_dataset.getTrainX()
            Y_train = pokemon_dataset.getTrainY()
            X_test = pokemon_dataset.getTestX()
            Y_test = pokemon_dataset.getTestY()

            kmeans = KMeans(400, max_iter=100)

            # Train the KMeans model
            kmeans.lloyd(X_train, tol=1e-7)

            #kmeans.projection(X=X_train, Y=Y_train)

            kmeans.save_weights()

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
        X_pca = X_train.reshape(X_train.shape[0], -1)

        optimal_components = PCA.determine_optimal_components(X_pca, variance_threshold=0.95)
        print(f"Optimal number of components to retain 95% variance: {optimal_components}")

        pcap = PCA(n_components=optimal_components)

        X_reduced = pcap.compress(X_pca)
        X_reconstructed = pcap.decompress(X_reduced)

        fig, axes = plt.subplots(2, 10, figsize=(15, 4))
        fig.suptitle('Images avant et après compression/décompression')

        for i in range(10):
            if dataset_choice == 'pokemon':
                axes[0, i].imshow(np.clip(X_pca[i].reshape(*dataset.image_size, 3) / 255.0, 0, 1))  # Normalize to [0, 1]
                axes[1, i].imshow(np.clip(X_reconstructed[i].reshape(*dataset.image_size, 3) / 255.0, 0, 1))
            else:
                axes[0, i].imshow(X_pca[i].reshape(dataset.image_size), cmap='gray')
                axes[1, i].imshow(X_reconstructed[i].reshape(dataset.image_size), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].axis('off')

        plt.show()

        clusters = pcap.cluster_with_pca(X_reduced, Y_train)
        pcap.visualize_clusters(X_reduced, clusters)

        # Fix plot_sample_images to handle RGB images and ensure valid indices
        def plot_sample_images(X, clusters, n_clusters=10):
            fig, axes = plt.subplots(n_clusters, 10, figsize=(15, 15))
            for cluster in range(n_clusters):
                cluster_indices = np.where(clusters == cluster)[0]
                for i, ax in enumerate(axes[cluster]):
                    if i < len(cluster_indices):
                        if dataset_choice == 'pokemon':
                            ax.imshow(np.clip(X[cluster_indices[i]].reshape(*dataset.image_size, 3) / 255.0, 0, 1))
                        else:
                            ax.imshow(X[cluster_indices[i]].reshape(dataset.image_size), cmap='gray')
                    ax.axis('off')
            plt.show()

        plot_sample_images(X_reconstructed, clusters, n_clusters=10)

        pcap.find_cluster_centers(X_reduced, n_clusters=10)
        X_generated = pcap.generate_image(n_images=5)

        if dataset_choice == 'pokemon':
            plt.imshow(np.clip(X_generated[0].reshape(*dataset.image_size, 3) / 255.0, 0, 1))
        else:
            plt.imshow(X_generated[0].reshape(dataset.image_size), cmap='gray')
        plt.title('Generated Image')
        plt.show()

        # Perform and visualize latent space walk
        pcap.latent_space_walk(X_pca, Y_train, n_steps=10)

        # Create a new PCA instance for 2D visualization
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.compress(X_pca)
        pca_2d.visualize_latent_space_distribution(X_pca_2d, Y_train)

        # 3D Visualization
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.compress(X_pca)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=Y_train, cmap='viridis')
        ax.set_title('3D PCA Visualization of Clusters')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        plt.show()

    elif model_choice == 'autoencoder':
        dataset_choice = input("Enter the dataset to run ('mnist' or 'pokemon'): ").strip().lower()

        if "mnist" == dataset_choice:
            ae = AEMnist(2, (784,), 784)
            ae.build()
            Xae_train, Xae_test = ae.standardize(X_train.copy(), X_test.copy())
            print(Xae_train.shape)
            ae.project_losses(ae.fit(Xae_train, Xae_train, 10_000, 32))
            ae.projection(Xae_train, ae.autoencoder.predict(Xae_train), Y_train, graph=False)
            ae.projection(Xae_train, ae.autoencoder.predict(Xae_train), Y_train, graph=True)
            ae.generation()
        elif "pokemon" == dataset_choice:
            ae = AEPokemon(3, (32*32*3), 32*32*3)
            ae.build_model()
            Xae_train, Xae_test = ae.normalize(X_train_pokemon.copy(), X_test_pokemon.copy())
            print(Xae_train.shape)
            ae.fit(Xae_train, Xae_train, 20_000, 32)
            predict_results = ae.autoencoder.predict(Xae_train)
            ae.projection_image(Xae_train, predict_results, size=32)
            ae.projection_3d(predict_results, Y_train_pokemon)
            ae.generation(size=32, n=15, linsize=30)
            ae.save_model("autoencoder/pokemon/models/pokemon_autoencoder.h5")

    elif model_choice == 'vae':
        dataset_choice = input("Enter the dataset to run ('mnist' or 'pokemon'): ").strip().lower()

        if "mnist" == dataset_choice:
            vae = VAEMnist(2, 28*28, 28*28)
            vae.build()
            X_vae_train, X_vae_test = vae.standardize(X_train.copy(), X_test.copy())
            vae.fit(X_vae_train, X_vae_train, 100_000, 32)
            X_train_encoded = vae.compression(X_vae_train)
            vae.projection(X_vae_train, X_train_encoded, Y_train, graph=False, n=20)
            vae.projection(X_vae_train, X_train_encoded, Y_train, graph=True)
            vae.generation()
        elif "pokemon" == dataset_choice:
            vae = VAEPokemon(3, 32*32*3, 32*32*3)
            vae.build_model()
            X_vae_train, X_vae_test = vae.normalize(X_train_pokemon.copy(), X_test_pokemon.copy())
            print(X_vae_train.shape)
            vae.fit(X_vae_train, X_vae_train, 50_000, 512)
            predict_results = vae.compression(X_vae_train)
            vae.projection_image(X_vae_train, vae.decrompression(predict_results), size=32)
            vae.projection_3d(predict_results, Y_train_pokemon)
            vae.generation(size=32, n=15, linsize=30)

    elif model_choice == 'som':
        map_size = 5 * sqrt(X_train.shape[0])
        print(f"Map size: {map_size}")
        # Find the multiplication of map size, for example if map size is 64, the x and y will be 8*8
        x = int(sqrt(map_size))
        y = int(sqrt(map_size))
        input_size = 784
        som = SOM(x, y, input_size, num_epochs=20, learning_rate=0.02, NW=2)

        X_train = X_train.reshape(X_train.shape[0], -1) / 255
        som.train(X_train)

        som.save_weights()

        mapped_data = []
        labels = np.empty((x, y), dtype=object)
        for i, input_data in enumerate(X_train):
            med = som.min_euc_distance(input_data)
            mapped_data.append(list(med))
            labels[med[0]][med[1]] = Y_train[i]

        print("Labels: ", labels)

        plt.figure(figsize=(10, 8))
        plt.imshow(labels.astype('float64'), cmap='viridis', interpolation='nearest')
        plt.title("2D Representation of Iris Dataset using SOM")
        plt.xticks(np.arange(x))
        plt.yticks(np.arange(y))
        plt.colorbar(label='Iris Class')
        plt.show()

        exe = X_train[0]

        plt.imshow(exe.reshape(28, 28), cmap='gray')
        plt.show()

        exe = exe.reshape(1, -1) / 255

        med = som.min_euc_distance(exe)
        pred = labels[med[0]][med[1]]

        print(f"Predicted label: {pred}")

        #show the image of the predicted label
        plt.imshow(X_train[np.where(Y_train == pred)[0][0]].reshape(28, 28), cmap='gray')
        plt.show()

    else:
        print("Invalid model choice. Please enter 'kmeans' or 'pca' or 'autoencoder' or 'vae' or 'SOM'.")
