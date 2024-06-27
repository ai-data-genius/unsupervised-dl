import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import math
import os
from dataset.dataset_mnist import mnistData
from dataset.dataset_pokemon import PokemonData


class Discriminator(Model):
    def __init__(self, in_features, out_features):
        super(Discriminator, self).__init__()
        self.fc1 = layers.Dense(128)
        self.leaky_relu1 = layers.LeakyReLU(alpha=0.2)
        self.fc2 = layers.Dense(64)
        self.leaky_relu2 = layers.LeakyReLU(alpha=0.2)
        self.fc3 = layers.Dense(32)
        self.leaky_relu3 = layers.LeakyReLU(alpha=0.2)
        self.fc4 = layers.Dense(out_features)
        self.dropout = layers.Dropout(0.3)

    def call(self, x, training=False):
        x = tf.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.dropout(x, training=training)
        x = self.fc3(x)
        x = self.leaky_relu3(x)
        x = self.dropout(x, training=training)
        x = self.fc4(x)
        return x

class Generator(Model):
    def __init__(self, in_features, out_features):
        super(Generator, self).__init__()
        self.fc1 = layers.Dense(32)
        self.relu1 = layers.LeakyReLU(alpha=0.2)
        self.fc2 = layers.Dense(64)
        self.relu2 = layers.LeakyReLU(alpha=0.2)
        self.fc3 = layers.Dense(128)
        self.relu3 = layers.LeakyReLU(alpha=0.2)
        self.fc4 = layers.Dense(out_features)
        self.dropout = layers.Dropout(0.3)
        self.tanh = layers.Activation('tanh')
        self.bn = layers.BatchNormalization()

    def call(self, x, training=True):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x, training=training)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout(x, training=training)
        x = self.fc4(x)
        x = self.bn(x, training=training)
        x = self.tanh(x)
        return x

class DiscriminatorConv(Model):
    def __init__(self):
        super(DiscriminatorConv, self).__init__()
        self.conv1 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(300, 300, 3))
        self.leaky_relu1 = layers.LeakyReLU(alpha=0.2)
        self.dropout1 = layers.Dropout(0.3)
        self.conv2 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.leaky_relu2 = layers.LeakyReLU(alpha=0.2)
        self.dropout2 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1)

    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

class GeneratorConv(Model):
    def __init__(self, z_size):
        super(GeneratorConv, self).__init__()
        self.z_size = z_size
        self.fc1 = layers.Dense(64 * 75 * 75, input_shape=(z_size,))
        self.relu1 = layers.ReLU()
        self.reshape = layers.Reshape((75, 75, 64))
        self.conv1 = layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same')
        self.relu2 = layers.ReLU()
        self.conv2 = layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same')
        self.relu3 = layers.ReLU()
        self.conv3 = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh')

    def call(self, x, training=True):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.relu3(x)
        x = self.conv3(x)
        return x



class GAN:
    def __init__(self, z_size):
        self.z_size = z_size

    def real_loss(self, d_out, loss_fn):
        labels = tf.ones_like(d_out)
        loss = loss_fn(labels, d_out)
        return loss

    def fake_loss(self, d_out, loss_fn):
        labels = tf.zeros_like(d_out)
        loss = loss_fn(labels, d_out)
        return loss

    @tf.function
    def train_step(self, real_images, generator, discriminator, g_optim, d_optim, loss_fn):
        z = tf.random.uniform(minval=-1, maxval=1, shape=(real_images.shape[0], 100))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = generator(z, training=True)

            real_logits = discriminator(real_images, training=True)
            fake_logits = discriminator(fake_images, training=True)

            gen_loss = loss_fn(tf.ones_like(fake_logits), fake_logits)
            disc_loss_real = loss_fn(tf.ones_like(real_logits), real_logits)
            disc_loss_fake = loss_fn(tf.zeros_like(fake_logits), fake_logits)
            disc_loss = disc_loss_real + disc_loss_fake

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        g_optim.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        d_optim.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return disc_loss, gen_loss

    def train_mnist_gan(self, d, g, d_optim, g_optim, loss_fn, dataset, n_epochs, device, verbose=False):
        print(f'Training on [{device}]...')

        fixed_z = tf.random.uniform(minval=-1, maxval=1, shape=(16, 100))
        fixed_samples = []
        d_losses = []
        g_losses = []

        with tf.device(device):
            for epoch in range(n_epochs):
                print(f'Epoch [{epoch + 1}/{n_epochs}]:')
                d_running_batch_loss = 0
                g_running_batch_loss = 0
                for curr_batch, real_images in enumerate(dataset):
                    real_images = tf.cast(real_images, tf.float32)
                    real_images = (real_images / 127.5) - 1

                    disc_loss, gen_loss = self.train_step(real_images, g, d, g_optim, d_optim, loss_fn)

                    d_running_batch_loss += disc_loss
                    g_running_batch_loss += gen_loss

                    if curr_batch % 400 == 0 and verbose:
                        print(
                            f'\tBatch [{curr_batch:>4}/{len(dataset):>4}] - d_batch_loss: {disc_loss.numpy():.6f}\tg_batch_loss: {gen_loss.numpy():.6f}')

                d_epoch_loss = d_running_batch_loss / len(dataset)
                g_epoch_loss = g_running_batch_loss / len(dataset)
                d_losses.append(d_epoch_loss.numpy())
                g_losses.append(g_epoch_loss.numpy())

                print(f'epoch_d_loss: {d_epoch_loss.numpy():.6f} \tepoch_g_loss: {g_epoch_loss.numpy():.6f}')

                fixed_samples.append(g(fixed_z, training=False).numpy())

            with open('fixed_samples_pokemon.pkl', 'wb') as f:
                pkl.dump(fixed_samples, f)

        return d_losses, g_losses


def display_images(images, n_cols=4, figsize=(12, 6)):
    """
    Utility function to display a collection of images in a grid

    Parameters
    ----------
    images: Tensor
            tensor of shape (batch_size, channel, height, width)
            containing images to be displayed
    n_cols: int
            number of columns in the grid

    Returns
    -------
    None
    """
    plt.style.use('ggplot')
    n_images = len(images)
    n_rows = math.ceil(n_images / n_cols)
    plt.figure(figsize=figsize)
    for idx in range(n_images):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        image = images[idx]
        # make dims H x W x C
        image = image.permute(1, 2, 0)
        cmap = 'gray' if image.shape[2] == 1 else plt.cm.viridis
        ax.imshow(image, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def show_generated_images(epoch, n_cols=8):
    # Load fixed samples generated during training
    with open('fixed_samples_pokemon.pkl', 'rb') as f:
        fixed_samples = pkl.load(f)

    # Select samples generated at the specified epoch
    epoch_data = fixed_samples[epoch]

    # Determine the dimensions of the images
    batch_size, vector_size = epoch_data.shape  # Shape of each image (16, 784)
    height = int(np.sqrt(vector_size))  # Assuming square images
    width = int(np.sqrt(vector_size))  # Assuming square images

    # Plot the images
    plt.figure(figsize=(15, 15))
    for i in range(batch_size):
        plt.subplot(batch_size // n_cols, n_cols, i + 1)
        plt.imshow(epoch_data[i].reshape((height, width)), cmap='gray')  # Reshape vector to image dimensions
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_generated_images_conv(epoch, n_cols=8):
    # Load fixed samples generated during training
    with open('fixed_samples_pokemon.pkl', 'rb') as f:
        fixed_samples = pkl.load(f)

    # Select samples generated at the specified epoch
    epoch_data = fixed_samples[epoch]

    # Plot the images
    plt.figure(figsize=(15, 15))
    for i in range(len(epoch_data)):
        plt.subplot(len(epoch_data) // n_cols, n_cols, i + 1)
        plt.imshow((epoch_data[i] * 0.5 + 0.5).astype(np.float32))# Scale images back to [0, 1]
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Instancier le Discriminateur et le Générateur

# Chemin vers votre dataset Pokémon
dataset_path = '../dataset/pokemon/00_Normal'

# Créer un dataset à partir des fichiers d'images
def load_dataset(dataset_path):
    images = []
    for filename in os.listdir(dataset_path):
        img = tf.keras.preprocessing.image.load_img(
            os.path.join(dataset_path, filename),
            target_size=(300, 300)  # Redimensionner toutes les images à 300x300
        )
        img = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img)
    return np.array(images)

# Charger les images
# pokemon_images = load_dataset(dataset_path)
#
# # Normalisation des images vers l'intervalle [-1, 1]
# pokemon_images = (pokemon_images.astype('float32') - 127.5) / 127.5
#
# # Créer un dataset TensorFlow à partir des images
# dataset = tf.data.Dataset.from_tensor_slices(pokemon_images)
#
# # Mélanger et diviser en lots
# batch_size = 16
# dataset = dataset.shuffle(buffer_size=len(pokemon_images)).batch(batch_size)

mnist = mnistData()
pokemon = PokemonData()
# X = mnist.getTrainX()
# y = mnist.getTrainY()
X = pokemon.getTrainX()
y = pokemon.getTrainY()
# Supposons que images et labels soient déjà définis
images = X
labels = y

images = images.astype('float32')


# Créez un dataset à partir des tuples (images, labels)
dataset = tf.data.Dataset.from_tensor_slices((images, labels))

# Mélanger les données
dataset = dataset.shuffle(buffer_size=len(images))

# Diviser en lots
dataset = dataset.batch(64)

# Précharger les données
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

gan = GAN(100)

d = Discriminator(in_features=784, out_features=1)
g = Generator(in_features=100, out_features=784)
# d = DiscriminatorConv()
# g = GeneratorConv(z_size=100)

# Afficher les modèles
print(d)
print()
print(g)

# Instancier les optimiseurs
d_optim = optimizers.Adam(learning_rate=0.002)
g_optim = optimizers.Adam(learning_rate=0.002)

# Instancier la fonction de perte
loss_fn = losses.BinaryCrossentropy(from_logits=True)

# Configurer le device
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

# Entraîner le modèle
n_epochs = 10
d_losses, g_losses = gan.train_mnist_gan(d, g, d_optim, g_optim, loss_fn, dataset, n_epochs, device, verbose=False)

plt.plot(d_losses, label='Discriminator')
plt.plot(g_losses, label='Generator')
plt.legend()
plt.show()

show_generated_images_conv(epoch=1, n_cols=8)

show_generated_images_conv(epoch=3, n_cols=8)

show_generated_images_conv(epoch=6, n_cols=8)

show_generated_images_conv(epoch=9, n_cols=8)



# show_generated_images(epoch=200, n_cols=8)
#
# show_generated_images(epoch=400, n_cols=8)
#
# show_generated_images(epoch=600, n_cols=8)
#
# show_generated_images(epoch=800, n_cols=8)
#
# show_generated_images(epoch=999, n_cols=8)