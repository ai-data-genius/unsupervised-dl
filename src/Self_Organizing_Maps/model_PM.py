import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class SOM:
    def __init__(self,X_train, x, y, input_size, learning_rate=0.2, NW=1.0, num_epochs=100, colors=False):
        self.input_size = input_size
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.NW = NW  # Neighbourhood width
        self.num_epochs = num_epochs
        self.colors = colors

        # Ensure X_train has enough samples to select from
        assert len(X_train) >= x * y, "The dataset doesn't have enough samples to initialize weights."

        # Randomly select images from the dataset for the weights
        selected_indices = np.random.choice(len(X_train), x * y, replace=False)
        selected_images = X_train[selected_indices].astype(np.float32)

        if self.colors:
            self.weights = selected_images.reshape(x, y, 3 * input_size)
            print(self.weights.shape)
        else:
            self.weights = selected_images.reshape(x, y, input_size)

    def min_euc_distance(self, input_data):
        # if self.colors:
        #     distances = np.linalg.norm(self.weights - input_data.reshape(1, 1, 3, self.input_size), axis=-1)
        # else:
        distances = np.linalg.norm(self.weights - input_data, axis=-1)
        med = np.unravel_index(np.argmin(distances), distances.shape)
        return med

    def update_weights(self, med, input_data):
        # Create a grid of coordinates
        x_indices, y_indices = np.meshgrid(np.arange(self.x), np.arange(self.y), indexing='ij')
        distances = np.sqrt((x_indices - med[0]) ** 2 + (y_indices - med[1]) ** 2)
        influences = np.exp(-np.power(distances, 2) / (2 * self.NW))

        # if self.colors:
        #     influences = influences[:, :, np.newaxis, np.newaxis]
        #     self.weights += self.learning_rate * influences * (input_data.reshape(1, 1, 3, self.input_size) - self.weights)
        # else:
        self.weights += self.learning_rate * influences[:, :, np.newaxis] * (input_data - self.weights)

    def process_input(self, input_data):
        med = self.min_euc_distance(input_data)
        self.update_weights(med, input_data)

    def train(self, data):
        for epoch in tqdm(range(self.num_epochs), desc="Training epochs", unit="epoch"):
            np.random.shuffle(data)  # Shuffle data in-place
            list(map(self.process_input, tqdm(data, desc=f"Epoch {epoch + 1}", unit="data point", leave=False)))

            fig, axs = plt.subplots(self.x, self.y, figsize=(self.y, self.x))
            for i in range(self.x):
                for j in range(self.y):
                    normalized_weights = (self.weights[i, j] - self.weights[i, j].min()) / (
                                self.weights[i, j].max() - self.weights[i, j].min())
                    axs[i, j].imshow(normalized_weights.reshape(32, 32, 3))
                    axs[i, j].axis('off')
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            file_path = f'model_{self.x}_{self.y}_{self.NW}_{self.num_epochs}_{self.learning_rate}'
            plt.savefig(f'images/{file_path}_epoch_{epoch + 1 + 300}.png')
            plt.close(fig)
        return self.weights

    def save_weights(self):
        np.save(f'models/model_{self.x}_{self.y}_{self.NW}_{self.num_epochs}_{self.learning_rate}', self.weights)

    def load_weights(self, file_path):
        self.weights = np.load(file_path)

    def compression(self, data):
        compressed_data = []
        for input_data in data:
            med = self.min_euc_distance(input_data)
            compressed_data.append(list(med))
        return compressed_data

    def decompression(self, compressed_data):
        decompressed_data = []
        for data in compressed_data:
            decompressed_data.append(self.weights[data[0], data[1]])
        return decompressed_data
