from keras.datasets import mnist
import tensorflow

class mnistData:

    def __init__(self):
        (self.train_X, self.train_y), (self.test_X, self.test_y) = mnist.load_data()

    def show(self):
        print('X_train: ' + str(self.train_X.shape))
        print('Y_train: ' + str(self.train_y.shape))
        print('X_test:  ' + str(self.test_X.shape))
        print('Y_test:  ' + str(self.test_y.shape))

    def getTrainX(self):
        return self.train_X

    def getTrainY(self):
        return self.train_y

    def getTestX(self):
        return self.test_X

    def getTestY(self):
        return self.test_y