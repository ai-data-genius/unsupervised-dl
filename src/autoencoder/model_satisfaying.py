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
        self.encoder.add(layers.Dense(512))
        self.encoder.add(layers.Activation('relu'))
        self.encoder.add(layers.Dense(256))
        self.encoder.add(layers.Activation('relu'))
        self.encoder.add(layers.Dense(128))
        self.encoder.add(layers.Activation('relu'))
        self.encoder.add(layers.Dense(32, activation="sigmoid"))
        self.encoder.add(layers.Dense(2, activation="sigmoid"))


        #sequential model #decoder
        self.decoder = keras.models.Sequential()
        self.decoder.add(layers.InputLayer((2,)))
        self.decoder.add(layers.Dense(32, activation="sigmoid"))
        self.decoder.add(layers.Dense(128))
        self.decoder.add(layers.Activation('relu'))
        self.decoder.add(layers.Dense(256))
        self.decoder.add(layers.Activation('relu'))
        self.decoder.add(layers.Dense(512))
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
        self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test))


    def compression(self: 'Kmeans') -> None:
        pass

    def decrompression(self: 'Kmeans') -> None:
        pass

    def projection(self, X_test, predict_results) -> None:

        n = 10  # How many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(X_test[i].reshape(28, 28))
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

    def generation(self: 'Kmeans') -> None:
        pass
