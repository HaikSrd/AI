import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, dataset, labels):
        self.labels = np.array(labels).reshape(-1,1)
        self.dataset = np.array(dataset)
        if self.dataset.ndim == 1:
            self.dataset = np.array([self.dataset])
        if self.dataset.shape[0] != self.labels.shape[0]:
            self.dataset = self.dataset.T
        print(self.dataset.shape)
        self.num_data = self.labels.shape[0]
        self.w0 = 0
        self.w1 = np.zeros((self.dataset.shape[1], 1))

    def loss(self):
        pred = self.dataset @ self.w1 + self.w0
        return np.mean((pred - self.labels)**2).astype('float64')

    def dw(self, w0, w1):
        sum_w1 = np.zeros((self.dataset.shape[1], 1))
        sum_w0 = 0
        for i in range(self.num_data):
            pred = self.dataset[i] @ w1 + w0
            error = pred - self.labels[i]
            sum_w0 += error
            sum_w1 += self.dataset[i].reshape(-1,1) * error
        return sum_w0/self.num_data, sum_w1/self.num_data

    def train(self, learning_rate = 0.001, epoch = 1000):
        """
        The cost function we use for linear regression is: (sum(y_true - w1x - w0)^2)/2
        """
        for i in range(epoch):
            dw0, dw1 = self.dw(self.w0, self.w1)
            self.w1 -= learning_rate*dw1
            self.w0 -= learning_rate*dw0
            if i % int(epoch/10) == 0:
                print(f"Epoch = {i}, Loss = {self.loss()}")
        return self.w1, self.w0

    def plot(self, weight= 0, bias = 0):
        if self.dataset.shape[1] != 1:
            raise ValueError("The plot function only works for linear polynomials")
        x = self.dataset
        y = self.labels.flatten()
        plt.scatter(x,y, color = 'blue')
        x_line = np.linspace(min(x), max(x), 100)
        y_line = weight * x_line + bias
        plt.plot(x_line, y_line, color = 'red')
        plt.show()

