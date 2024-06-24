from kmeans.model import KMeans
from dataset.dataset_mnist import mnistData
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    mnist = mnistData()
    X = mnist.getTrainX()

    #initialize the model
    kmeans = KMeans(10, max_iter=20)
    kmeans.lloyd(X)

    #plot the clusters
    kmeans.plot_clusters(X)