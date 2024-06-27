import os

import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class PokemonData:
    def __init__(self, image_size=(64, 64), test_size=0.2, random_state=42):
        self.dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/pokemon/')
        self.image_size = image_size
        self.test_size = test_size
        self.random_state = random_state
        self.train_X, self.test_X, self.train_y, self.test_y, self.class_names = self._create_train_test_split()

    def _load_images_and_labels(self):
        images = []
        labels = []
        class_names = []

        for label, class_name in enumerate(os.listdir(self.dataset_dir)):
            class_dir = os.path.join(self.dataset_dir, class_name)

            if os.path.isdir(class_dir):
                class_names.append(class_name)

                for filename in os.listdir(class_dir):
                    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        img_path = os.path.join(class_dir, filename)
                        img = Image.open(img_path).convert('RGB')
                        img_resized = img.resize(self.image_size, Image.Resampling.LANCZOS)
                        images.append(np.array(img_resized))
                        labels.append(label)

        return np.array(images), np.array(labels), class_names

    def _create_train_test_split(self):
        images, labels, class_names = self._load_images_and_labels()
        x_train, x_test, y_train, y_test = train_test_split(
            images, labels,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        return x_train, x_test, y_train, y_test, class_names

    def show(self):
        print(f"X_train: {self.train_X.shape}")
        print(f"Y_train: {self.train_y.shape}")
        print(f"X_test: {self.test_X.shape}")
        print(f"Y_test: {self.test_y.shape}")
        print(f"Classes: {self.class_names}")

    def getTrainX(self):
        return self.train_X

    def getTrainY(self):
        return self.train_y

    def getTestX(self):
        return self.test_X

    def getTestY(self):
        return self.test_y

    def show_images(self):
        plt.figure(figsize=(10, 10))
        for i in range(0,len(self.train_X)):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_X[i])
            plt.xlabel(self.class_names[self.train_y[i] - 1])
        plt.show()
        plt.figure(figsize=(10, 10))
        for i in range(0, len(self.test_X)):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.test_X[i])
            plt.xlabel(self.class_names[self.test_y[i] - 1])
        plt.show()
