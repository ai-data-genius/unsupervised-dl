import matplotlib.pyplot as plt
import numpy as np

from keras import layers, Model, Input, backend as K
from keras.losses import binary_crossentropy


class AE:
    def __init__(self: 'AE', latent_dim, input_shape, output_shape):
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build(self: 'AE') -> None:
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
        vae_loss = K.mean(reconstruction_loss + kl_loss)

        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')

    def standardize(self: 'AE', X_train, X_test):
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.
        X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
        X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

        return X_train, X_test

    def fit(self: 'AE', X_train, X_test, epochs, batch_size) -> None:
        history = self.vae.fit(
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_test, None),
        )

        # Plotting the training and validation loss
        plt.figure(figsize=(10, 5))

        ## Plotting training loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        ## Plotting validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.show()

    def compression(self, X) -> None:
        return self.encoder.predict(X, batch_size=32)[0]  # Use z_mean

    def decrompression(self, X) -> None:
        return self.decoder.predict(X)

    def projection(self, X, predict_results, y_test, graph=False, n=10) -> None:
        if not graph:
            decoded_imgs = self.decoder.predict(predict_results)
            plt.figure(figsize=(20, 4))

            for i in range(n):
                # Display original
                ax = plt.subplot(2, n, i + 1)
                plt.imshow(X[i].reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # Display reconstruction
                ax = plt.subplot(2, n, i + 1 + n)
                plt.imshow(decoded_imgs[i].reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            plt.show()
        else:
            plt.figure(figsize=(6, 6))
            plt.scatter(predict_results[:, 0], predict_results[:, 1], c=y_test, cmap='viridis')
            plt.colorbar()
            plt.show()


    def generation(self) -> None:
        n = 15
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        grid_x = np.linspace(-3, 3, n)
        grid_y = np.linspace(-3, 3, n)

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[
                    i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size,
                ] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='viridis')
        plt.show()
