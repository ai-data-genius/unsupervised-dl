import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

class AE:
    def __init__(self, encoding_dim, input_shape, output_shape):
        self.encoder = None
        self.encoding_dim = encoding_dim
        self.decoder = None
        self.input_shape = input_shape
        self.autoencoder = None
        self.output_shape = output_shape

    def build(self: 'AE') -> None:
        #sequential model #encoder
        self.encoder = keras.models.Sequential()
        self.encoder.add(layers.InputLayer(self.input_shape))
        self.encoder.add(layers.Flatten())
        self.encoder.add(layers.BatchNormalization())
        self.encoder.add(layers.Dense(512, activation="tanh"))
        self.encoder.add(layers.Dense(256, activation="tanh"))
        self.encoder.add(layers.Dense(128, activation="tanh"))
        self.encoder.add(layers.Dense(32, activation="sigmoid"))
        self.encoder.add(layers.BatchNormalization())
        self.encoder.add(layers.Dense(self.encoding_dim, activation="sigmoid"))


        #sequential model #decoder
        self.decoder = keras.models.Sequential()
        self.decoder.add(layers.InputLayer((self.encoding_dim,)))
        self.decoder.add(layers.BatchNormalization())
        self.decoder.add(layers.Dense(32, activation="tanh"))
        self.decoder.add(layers.Dense(128, activation="tanh"))
        self.decoder.add(layers.Dense(256, activation="tanh"))
        self.decoder.add(layers.Dense(512, activation="tanh"))
        self.decoder.add(layers.Dense(self.output_shape, activation='sigmoid'))

        self.autoencoder = keras.models.Sequential([self.encoder, self.decoder])

    def standardize(self: 'AE', X_train, X_test):
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.
        X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
        X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

        return X_train, X_test

    def fit (self: 'AE', X_train, X_test, epochs, batch_size) -> None:
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        history = self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                                       validation_data=(X_test, X_test))

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

        #save the model
        keras.saving.save_model(self.autoencoder, '../models_AE/autoencoder1.keras')
    def compression(self, X) -> None:
        return self.encoder.predict(X)

    def decrompression(self, X) -> None:
        return self.decoder.predict(X)

    def projection(self, X, predict_results, y_test, graph=False, n=10) -> None:
        if not graph:
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
                plt.imshow(predict_results[i].reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()
        else:
            plt.figure(figsize=(6, 6))
            plt.scatter(predict_results[:, 0], predict_results[:, 1], c=y_test)
            plt.colorbar()
            plt.show()


    def generation(self) -> None:
        n = 15  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # We will sample n points within [-15, 15] standard deviations
        grid_x = np.linspace(-15, 15, n)
        grid_y = np.linspace(-15, 15, n)

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure)
        plt.show()
