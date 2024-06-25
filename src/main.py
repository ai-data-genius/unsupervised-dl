from kmeans.model import KMeans
from dataset.dataset_mnist import mnistData
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    mnist = mnistData()
    X = mnist.getTrainX()
    Y = mnist.getTrainY()

    #initialize the model
    kmeans = KMeans(10, max_iter=100)
    kmeans.lloyd(X)

    #plot the clusters
    kmeans.plot_clusters(X, Y)

    #generate a new image
    X_gen = kmeans.generate(1, 1)

    plt.imshow(X_gen[0].reshape(28, 28), cmap='gray')
    plt.show()