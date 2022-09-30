import numpy as np
class Perceptron:
    def __init__(self):
        self.weights = None

    def training(self, X, y, l_r, epoch):
        s, f = X.shape
        self.weights = np.zeros(f)
        bias = 0
        for i in range(epoch):
            for j in range(s):
                update = l_r * (y[j] - self.predict(X[j], bias))
                self.weights += update * X[j]
                bias += update



    def predict(self, input, bias):
        out = np.dot(input, self.weights) + bias
        pred = self.act(out)
        return pred

    def act(self, input):
        return np.where(input >= 0, 1, -1)
