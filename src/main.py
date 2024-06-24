import numpy as np
import kmeans.model as model

#generate two clouds of points in 2D

np.random.seed(0)
n_samples = 1000
n_clusters = 2

X = np.random.randn(n_samples, 2)
X[:n_samples // 2] += 5
X[n_samples // 2:] += 0

#plot the data
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1])
#plt.show()

print(X.shape)


#initialize the model
kmeans = model.kmeans(n_clusters)
kmeans.lloyd(X)