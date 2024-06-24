from kmeans.model import kmeans
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    np.random.seed(np.random.randint(100))
    n_samples = 1000
    n_clusters = 3

    X = np.random.randn(n_samples, 2)
    X[:300] += 5
    X[300:600] += 10
    X[600:] += 15

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    print(X.shape)


    #initialize the model
    kmeans = kmeans(n_clusters)
    kmeans.lloyd(X)

    #plot the clusters
    kmeans.plot_clusters(X)