import keras
import matplotlib.pyplot as plt
import numpy as np

from keras import layers, models
from mpl_toolkits.mplot3d import Axes3D


class AE:
    def __init__(self: "AE", encoding_dim, input_shape, output_shape):
        self.encoder = None
        self.encoding_dim = encoding_dim
        self.decoder = None
        self.input_shape = input_shape
        self.autoencoder = None
        self.output_shape = output_shape

    def set_encoder(self: "AE") -> None:
        self.encoder = models.Sequential()
        self.encoder.add(layers.InputLayer(self.input_shape))
        self.encoder.add(layers.Dense(2048, activation=keras.activations.tanh))
        self.encoder.add(layers.LayerNormalization())
        self.encoder.add(layers.Dense(1024, activation=keras.activations.tanh))
        self.encoder.add(layers.LayerNormalization())
        self.encoder.add(layers.Dense(512, activation=keras.activations.tanh))
        self.encoder.add(layers.LayerNormalization())
        self.encoder.add(layers.Dense(256, activation=keras.activations.tanh))
        self.encoder.add(layers.LayerNormalization())
        self.encoder.add(layers.Dense(128, activation=keras.activations.tanh))
        self.encoder.add(layers.LayerNormalization())
        self.encoder.add(layers.Dense(64, activation=keras.activations.tanh))
        self.encoder.add(layers.LayerNormalization())
        self.encoder.add(layers.Dense(32, activation=keras.activations.tanh))
        self.encoder.add(layers.LayerNormalization())
        self.encoder.add(layers.Dense(16, activation=keras.activations.tanh))
        self.encoder.add(layers.LayerNormalization())
        self.encoder.add(layers.Dense(self.encoding_dim, activation="sigmoid"))

    def set_decoder(self: "AE") -> None:
        self.decoder = models.Sequential()
        self.decoder.add(layers.InputLayer((self.encoding_dim,)))
        self.decoder.add(layers.Dense(16, activation=keras.activations.tanh))
        self.decoder.add(layers.LayerNormalization())
        self.decoder.add(layers.Dense(32, activation=keras.activations.tanh))
        self.decoder.add(layers.LayerNormalization())
        self.decoder.add(layers.Dense(64, activation=keras.activations.tanh))
        self.decoder.add(layers.LayerNormalization())
        self.decoder.add(layers.Dense(128, activation=keras.activations.tanh))
        self.decoder.add(layers.LayerNormalization())
        self.decoder.add(layers.Dense(256, activation=keras.activations.tanh))
        self.decoder.add(layers.LayerNormalization())
        self.decoder.add(layers.Dense(512, activation=keras.activations.tanh))
        self.decoder.add(layers.LayerNormalization())
        self.decoder.add(layers.Dense(1024, activation=keras.activations.tanh))
        self.decoder.add(layers.LayerNormalization())
        self.decoder.add(layers.Dense(2048, activation=keras.activations.tanh))
        self.decoder.add(layers.LayerNormalization())
        self.decoder.add(layers.Dense(self.output_shape, activation='sigmoid'))

    def build_model(self: "AE") -> None:
        self.set_encoder()
        self.set_decoder()
        self.autoencoder = keras.models.Sequential([self.encoder, self.decoder])

    def normalize(self: 'AE', X_train, X_test):
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.

        return (
            X_train.reshape((len(X_train), np.prod(X_train.shape[1:]))),
            X_test.reshape((len(X_test), np.prod(X_test.shape[1:]))),
        )

    def fit(
        self: "AE",
        X_train,
        X_test,
        epochs,
        batch_size,
        initial_learning_rate=1e-5,
        final_learning_rate=1e-6,
        decay_steps=10000,
    ):
        initial_learning_rate = initial_learning_rate
        final_learning_rate = final_learning_rate
        decay_steps = decay_steps
        decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / decay_steps)

        self.autoencoder.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=initial_learning_rate,
                    decay_steps=decay_steps,
                    decay_rate=decay_rate,
                    staircase=True,
                ),
                weight_decay=1e-5,
            ),
            loss=keras.losses.binary_crossentropy,
        )

        return self.autoencoder.fit(
            X_train,
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_test, X_test),
          )

    def project_losses(self: "AE", history) -> None:
        # Plotting the training and validation loss
        plt.figure(figsize=(10, 5))

        # Plotting training loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        # Plotting validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()

        plt.show()

    def compression(self: "AE", X) -> None:
        return self.encoder.predict(X)

    def decrompression(self: "AE", X) -> None:
        return self.decoder.predict(X)

    def projection_image(self: "AE", X, predict_results, size: int) -> None:
        num_images = len(predict_results)
        num_rows = (num_images + 4) // 5
        plt.figure(figsize=(20, 4 * num_rows))
        
        for i in range(num_images):
            # Display original images
            ax = plt.subplot(num_rows, 10, 2 * i + 1)
            plt.imshow(X[i].reshape(size, size, 3))
            plt.axis('off')

            # Display reconstructed images
            ax = plt.subplot(num_rows, 10, 2 * i + 2)
            plt.imshow(predict_results[i].reshape(size, size, 3))
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def projection_3d(self: "AE", predict_results, y_test) -> None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(predict_results[:, 0], predict_results[:, 1], predict_results[:, 2], c=y_test)
        plt.colorbar(scatter)
        plt.show()

    def generation(self: "AE", size: int, n: int, linsize: int) -> None:
        figure = np.zeros((size * n, size * n, 3))
        grid_x = np.linspace(-linsize, linsize, n)
        grid_y = np.linspace(-linsize, linsize, n)
        grid_z = np.linspace(-linsize, linsize, n)

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                for k, zi in enumerate(grid_z):
                    z_sample = np.array([[xi, yi, zi]])
                    x_decoded = self.decoder.predict(z_sample)
                    digit = x_decoded[0].reshape(size, size, 3)
                    figure[i * size: (i + 1) * size, j * size: (j + 1) * size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure)
        plt.show()

    def save_model(self: "AE", filepath: str) -> None:
        self.autoencoder.save(filepath)

    @classmethod
    def load_model(cls: "AE", filepath: str) -> "AE":
        autoencoder = keras.models.load_model(filepath)
        instance = cls(encoding_dim=None, input_shape=None, output_shape=None)
        instance.autoencoder = autoencoder
        instance.encoder = autoencoder.layers[0]
        instance.decoder = autoencoder.layers[1]

        return instance
