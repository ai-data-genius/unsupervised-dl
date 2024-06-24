import numpy as np
from src.kmeans.model import kmeans
from src.dataset.dataset_mnist import mnistData
from matplotlib import pyplot as plt

dataset = mnistData()

print(dataset.getTrainX()[0])

#show image
from matplotlib import pyplot as plt

plt.imshow(dataset.getTrainX()[0])
plt.show()