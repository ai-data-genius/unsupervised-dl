import numpy as np


class toyData:
    def __init__(self: "toyData", n_samples=1000):
        self.X = np.random.randn(n_samples, 2)
        self.X[:300] += 5
        self.X[300:600] += 10
        self.X[600:] += 15

    def getX(self: "toyData"):
        return self.X
