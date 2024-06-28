from src.dataset.dataset_mnist import mnistData
from model import DiffusionModel
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist_dataset = mnistData()

X_train = mnist_dataset.getTrainX()
Y_train = mnist_dataset.getTrainY()
X_test = mnist_dataset.getTestX()
Y_test = mnist_dataset.getTestY()



X_train = (X_train.astype(np.float32) - 127.5) / 127.5

X_train = np.expand_dims(X_train, axis=-1)

batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(60000).batch(batch_size)

device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
with tf.device(device):
    model = DiffusionModel()
    model.train_model(train_dataset, epochs=5)


def show_generated_samples(model, num_samples=16):
    samples = model.sample((num_samples, 28, 28, 1))
    samples = (samples + 1) / 2  # Rescale to [0, 1]

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1))
    for i in range(num_samples):
        axes[i].imshow(samples[i, :, :, 0], cmap='gray')
        axes[i].axis('off')
    plt.show()


show_generated_samples(model)
