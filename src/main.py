from kmeans.model import KMeans
from dataset.dataset_mnist import mnistData
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    mnist = mnistData()
    X = mnist.getTrainX()
    Y = mnist.getTrainY()

    #initialize the model
    kmeans = KMeans(50, max_iter=200)
    kmeans.lloyd(X)

    #plot the clusters
    kmeans.plot_clusters(X, Y)