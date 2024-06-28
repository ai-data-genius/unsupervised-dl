import keras
import matplotlib.pyplot as plt
import numpy as np

from keras import layers, Model, Input, backend as K
from keras.datasets import mnist
from keras.losses import binary_crossentropy


class VAE:
    def __init__(self: "VAE", latent_dim, input_shape, output_shape):
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build_model(
        self: "VAE",
        initial_learning_rate=16e-5,
        final_learning_rate=16e-6,
        decay_steps=10000,
    ) -> None:
        original_dim = self.input_shape
        intermediate_dim = 64

        # Encoder
        inputs = Input(shape=(original_dim,))
        h = layers.Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = layers.Dense(self.latent_dim)(h)
        z_log_sigma = layers.Dense(self.latent_dim)(h)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=0.1)
            return z_mean + K.exp(z_log_sigma) * epsilon

        z = layers.Lambda(sampling)([z_mean, z_log_sigma])
        self.encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

        # Decoder
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = layers.Dense(original_dim, activation='sigmoid')(x)
        self.decoder = Model(latent_inputs, outputs, name='decoder')

        # VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')

        # Losses
        reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + 0.01 * kl_loss)

        initial_learning_rate = initial_learning_rate
        final_learning_rate = final_learning_rate
        decay_steps = decay_steps
        decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / decay_steps)

        self.vae.add_loss(vae_loss)
        self.vae.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=initial_learning_rate,
                    decay_steps=decay_steps,
                    decay_rate=decay_rate,
                    staircase=True,
                ),
                weight_decay=1e-5,
            ),
        )

    def normalize(self: "VAE", X_train, X_test):
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.

        return (
            X_train.reshape((len(X_train), np.prod(X_train.shape[1:]))),
            X_test.reshape((len(X_test), np.prod(X_test.shape[1:]))),
        )

    def project_losses(self: "VAE", history) -> None:
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

    def fit(
        self: "VAE",
        X_train,
        X_test,
        epochs,
        batch_size,
    ) -> None:
        return self.vae.fit(
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_test, None),
        )

    def compression(self: "VAE", X) -> None:
        return self.encoder.predict(X, batch_size=32)[0]  # Use z_mean

    def decrompression(self: "VAE", X) -> None:
        return self.decoder.predict(X)

    def projection_image(self: "VAE", X, predict_results, size: int, max_images: int = 100) -> None:
        num_images = min(len(predict_results), max_images)
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

    def projection_3d(self: "VAE", predict_results, y_test) -> None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(predict_results[:, 0], predict_results[:, 1], predict_results[:, 2], c=y_test)
        plt.colorbar(scatter)
        plt.show()

    def generation(self: "VAE", size: int, n: int, linsize: int) -> None:
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

    def save_model(self: "VAE", filepath: str) -> None:
        self.vae.save(filepath)

    @classmethod
    def load_model(cls: "VAE", filepath: str) -> "VAE":
        vae = keras.models.load_model(filepath)
        instance = cls(encoding_dim=None, input_shape=None, output_shape=None)
        instance.autoencoder = vae
        instance.encoder = vae.layers[0]
        instance.decoder = vae.layers[1]

        return instance
