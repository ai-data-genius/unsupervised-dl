from kmeans.model import KMeans
from dataset.dataset_mnist import mnistData
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    mnist_dataset = mnistData()
    X = mnist_dataset.getTrainX()
    Y = mnist_dataset.getTrainY()

    #initialize the model
    kmeans = KMeans(20, max_iter=30)
    kmeans.lloyd(X)

    #plot the clusters
    kmeans.plot_clusters(X, Y)

    #plot first image of X
    plt.imshow(X[1])
    plt.show()

    compressed = kmeans.compress(X[1])
    #decompress the image
    kmeans.decompress(compressed)

    #generate a new image
    X_gen = kmeans.generate(steps=25)