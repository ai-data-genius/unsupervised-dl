from kmeans.model import kmeans
from dataset.dataset_mnist import mnistData
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    mnist = mnistData()
    X = mnist.getTrainX()

    #initialize the model
    kmeans = kmeans(10)
    kmeans.lloyd(X)

    #plot the clusters
    kmeans.plot_clusters(X)