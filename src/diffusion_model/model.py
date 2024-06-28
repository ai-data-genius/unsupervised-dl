import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Ensure TensorFlow is using GPU if available
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"


class DiffusionModel(tf.keras.Model):
    def __init__(self, timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.timesteps = timesteps

        self.encoder = models.Sequential([
            layers.InputLayer((28, 28, 1)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), strides=2, activation='relu', padding='same'),
        ])

        self.decoder = models.Sequential([
            layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, (3, 3), activation='tanh', padding='same')
        ])

    def call(self, x, t):
        noise = tf.random.normal(tf.shape(x)) * (1 - (t / self.timesteps))
        x = x + noise
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def sample(self, shape):
        x = tf.random.normal(shape)
        for t in reversed(range(self.timesteps)):
            t_tensor = tf.constant(t, dtype=tf.float32)
            x = self(x, t_tensor)
        return x

    def train_model(self, dataset, epochs=5, learning_rate=0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()

        @tf.function
        def train_step(images):
            t = tf.random.uniform((images.shape[0],), 0, self.timesteps, dtype=tf.float32)

            with tf.GradientTape() as tape:
                reconstructed = self(images, t)
                loss = loss_fn(images, reconstructed)

            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            return loss

        for epoch in range(epochs):
            for step, images in enumerate(dataset):
                loss = train_step(images)

                if step % 100 == 0:
                    print(f'Epoch {epoch + 1}, Step {step}, Loss: {loss.numpy():.4f}')
