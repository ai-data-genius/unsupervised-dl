import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class SOM:
    def __init__(self, x, y, input_size, learning_rate=0.2, NW=1.0, num_epochs=100):
        self.input_size = input_size
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.NW = NW  # Neighbourhood width
        self.num_epochs = num_epochs
        self.weights = np.random.rand(x, y, input_size)

    def min_euc_distance(self, input_data):
        distances = np.linalg.norm(self.weights - input_data, axis=-1)
        med = np.unravel_index(np.argmin(distances), distances.shape)
        return med

    def update_weights(self, med, input_data):
        # Create a grid of coordinates
        x_indices, y_indices = np.meshgrid(np.arange(self.x), np.arange(self.y), indexing='ij')

        distances = np.sqrt((x_indices - med[0]) ** 2 + (y_indices - med[1]) ** 2)

        influences = np.exp(-np.power(distances, 2) / (2 * self.NW))

        # Update the weights
        self.weights += self.learning_rate * influences[:, :, np.newaxis] * (input_data - self.weights)

    def process_input(self, input_data):
        med = self.min_euc_distance(input_data)
        self.update_weights(med, input_data)

    def train(self, data):
        for epoch in tqdm(range(self.num_epochs), desc="Training epochs", unit="epoch"):
            np.random.shuffle(data)  # Shuffle data in-place
            list(map(self.process_input, tqdm(data, desc=f"Epoch {epoch + 1}", unit="data point", leave=False)))
            # Display the weights as an image of a digit in matrix using self.x and self.y
            plt.imshow(self.weights.reshape(self.x, self.y, 28, 28).transpose(0, 2, 1, 3).reshape(self.x * 28, self.y * 28),
                          cmap='gray')
            plt.axis('off')
            file_path = f'model_{self.x}_{self.y}_{self.NW}_{self.num_epochs}_{self.learning_rate}'
            #save the png
            plt.savefig(f'Self_Organizing_Maps/images/{file_path}_epoch_{epoch + 1}.png')
        return self.weights

    def save_weights(self):
        np.save(f'Self_Organizing_Maps/models/model_{self.x}_{self.y}_{self.NW}_{self.num_epochs}_{self.learning_rate}', self.weights)

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