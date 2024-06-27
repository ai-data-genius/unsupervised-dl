import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm
from dataset.dataset_pokemon import PokemonData


class Discriminator(Model):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()
        self.fc1 = layers.Dense(1024, input_shape=(in_features,))
        self.leaky_relu1 = layers.LeakyReLU(alpha=0.2)
        self.fc2 = layers.Dense(512)
        self.leaky_relu2 = layers.LeakyReLU(alpha=0.2)
        self.fc3 = layers.Dense(256)
        self.leaky_relu3 = layers.LeakyReLU(alpha=0.2)
        self.fc4 = layers.Dense(128)
        self.leaky_relu4 = layers.LeakyReLU(alpha=0.2)
        self.fc5 = layers.Dense(64)
        self.leaky_relu5 = layers.LeakyReLU(alpha=0.2)
        self.fc6 = layers.Dense(32)
        self.leaky_relu6 = layers.LeakyReLU(alpha=0.2)
        self.fc7 = layers.Dense(1)

    def call(self, x, training=False):
        x = tf.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.fc3(x)
        x = self.leaky_relu3(x)
        x = self.fc4(x)
        x = self.leaky_relu4(x)
        x = self.fc5(x)
        x = self.leaky_relu5(x)
        x = self.fc6(x)
        x = self.leaky_relu6(x)
        x = self.fc7(x)
        return x


class Generator(Model):
    def __init__(self, out_features, z_size):
        super(Generator, self).__init__()
        self.z_size = z_size
        self.fc1 = layers.Dense(1024, input_shape=(self.z_size,))
        self.relu1 = layers.LeakyReLU(alpha=0.2)
        self.fc2 = layers.Dense(2048)
        self.relu2 = layers.LeakyReLU(alpha=0.2)
        self.fc3 = layers.Dense(4096)
        self.relu3 = layers.LeakyReLU(alpha=0.2)
        self.fc4 = layers.Dense(out_features, activation='tanh')  # Output layer with tanh activation

    def call(self, x, training=True):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        img = tf.reshape(x, (-1, 32, 32, 3))  # Reshape to (batch_size, 32, 32, 3)
        return img


class WGAN:
    def __init__(self, z_size):
        self.z_size = z_size

    def discriminator_loss(self, real_logits, fake_logits):
        return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

    def generator_loss(self, fake_logits):
        return -tf.reduce_mean(fake_logits)

    @tf.function
    def train_step(self, real_images, generator, discriminator, g_optim, d_optim):
        z = tf.random.uniform(minval=-1, maxval=1, shape=(real_images.shape[0], self.z_size))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = generator(z, training=True)

            real_logits = discriminator(real_images, training=True)
            fake_logits = discriminator(fake_images, training=True)

            disc_loss = self.discriminator_loss(real_logits, fake_logits)
            gen_loss = self.generator_loss(fake_logits)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        g_optim.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        d_optim.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return disc_loss, gen_loss

    def train_mnist_wgan(self, d, g, d_optim, g_optim, dataset, n_epochs, device, verbose=False):
        print(f'Training on [{device}]...')

        fixed_z = tf.random.uniform(minval=-1, maxval=1, shape=(16, self.z_size))
        fixed_samples = []
        d_losses = []
        g_losses = []

        for epoch in range(n_epochs):
            d_running_batch_loss = 0
            g_running_batch_loss = 0
            num_batches = 0

            for real_images in dataset:
                real_images = tf.cast(real_images, tf.float32)
                real_images = (real_images / 127.5) - 1

                disc_loss, gen_loss = self.train_step(real_images, g, d, g_optim, d_optim)

                d_running_batch_loss += disc_loss
                g_running_batch_loss += gen_loss
                num_batches += 1

            d_epoch_loss = d_running_batch_loss / num_batches
            g_epoch_loss = g_running_batch_loss / num_batches
            d_losses.append(d_epoch_loss.numpy())
            g_losses.append(g_epoch_loss.numpy())

            print(f'Epoch [{epoch + 1}/{n_epochs}], d_loss: {d_epoch_loss.numpy()}, g_loss: {g_epoch_loss.numpy()}')

            generated_samples = g(fixed_z, training=False).numpy()
            assert generated_samples.shape[
                       -1] == 3, f"Generated samples do not have 3 channels, but {generated_samples.shape[-1]}"
            fixed_samples.append(generated_samples)

            with open('fixed_samples_pokemon.pkl', 'wb') as f:
                pkl.dump(fixed_samples, f)

        return d_losses, g_losses

    def display_images(self, images):
        images = (images * 0.5 + 0.5).numpy()  # Rescale to [0, 1] and convert to numpy array
        plt.figure(figsize=(15, 15))
        for i in range(min(8, len(images))):
            plt.subplot(1, 8, i + 1)
            plt.imshow(images[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()


def show_generated_images(epoch, n_cols=8):
    # Load fixed samples generated during training
    with open('fixed_samples_pokemon.pkl', 'rb') as f:
        fixed_samples = pkl.load(f)

    # Select samples generated at the specified epoch
    epoch_data = fixed_samples[epoch]

    # Plot the images
    plt.figure(figsize=(15, 15))
    for i in range(len(epoch_data)):
        plt.subplot(len(epoch_data) // n_cols, n_cols, i + 1)
        plt.imshow((epoch_data[i] * 0.5 + 0.5).astype(np.float32))  # Scale images back to [0, 1]
        plt.axis('off')
    plt.tight_layout()
    plt.show()


pokemon = PokemonData(image_size=(32,32))
X = pokemon.getTrainX()
y = pokemon.getTrainY()
# Supposons que images et labels soient déjà définis
images = X.astype('float32')

# Créez un dataset à partir des tuples (images, labels)
dataset = tf.data.Dataset.from_tensor_slices(images)

# Mélanger les données
dataset = dataset.shuffle(buffer_size=len(images))

# Diviser en lots
dataset = dataset.batch(64)

# Précharger les données
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Instancier le Discriminateur et le Générateur
gan = WGAN(z_size=100)
d = Discriminator(in_features=3072)
g = Generator(out_features=3072, z_size=100)

# Instancier les optimiseurs avec des taux d'apprentissage plus bas
d_optim = optimizers.Adam(learning_rate=1e-5, beta_1=0.0, beta_2=0.9)
g_optim = optimizers.Adam(learning_rate=1e-5, beta_1=0.0, beta_2=0.9)

# Configurer le device
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

# Entraîner le modèle avec la fonction d'entraînement adaptée à WGAN
n_epochs = 10000
d_losses, g_losses = gan.train_mnist_wgan(d, g, d_optim, g_optim, dataset, n_epochs, device, verbose=False)

# Afficher les courbes d'apprentissage
plt.plot(d_losses, label='Discriminator')
plt.plot(g_losses, label='Generator')
plt.legend()
plt.show()

# Afficher des échantillons générés à différents moments de l'entraînement
for i in range(10000):
    if i % 1000 == 0:
        show_generated_images(epoch=i, n_cols=8)
