import numpy as np
from tqdm import tqdm


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

        influences = np.exp(-np.power(distances, 2) / 2 * self.NW)

        # Update the weights
        self.weights += self.learning_rate * influences[:, :, np.newaxis] * (input_data - self.weights)

    def process_input(self, input_data):
        med = self.min_euc_distance(input_data)
        self.update_weights(med, input_data)

    def train(self, data):

        #shuffle the data
        np.random.shuffle(data)

        for epoch in tqdm(range(self.num_epochs), desc="Training epochs", unit="epoch"):
            list(map(self.process_input, tqdm(data, desc=f"Epoch {epoch + 1}", unit="data point", leave=False)))

        return self.weights

    def save_weights(self):
        np.save(f'models/model_{self.x}_{self.y}_{self.NW}_{self.num_epochs}_{self.learning_rate}', self.weights)

    def load_weights(self, file_path):
        self.weights = np.load(file_path)