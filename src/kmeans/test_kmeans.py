import numpy as np
from src.kmeans.model import kmeans
from src.dataset.dataset_mnist import mnistData
#generate two clouds of points in 2D

n_samples = 1000
n_clusters = 3

X = np.random.randn(n_samples, 2)
X[:300] += 5
X[300:600] += 10
X[600:] += 15

#plot the data
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1])
plt.show()

print(X.shape)


#initialize the model
kmeans = kmeans(n_clusters)
kmeans.lloyd(X)

#plot the clusters
kmeans.plot_clusters(X)


dataset = mnistData()

