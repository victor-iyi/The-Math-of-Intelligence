"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 13 September, 2017 @ 12:24 AM.
  Copyright (c) 2017. Victor. All rights reserved.
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=1e-4):
        self.learning_rate = learning_rate
        self.W = np.array([])
        self.b = np.array([])

    def fit(self, X, y, num_iter=10000):
        weight_shape = [X.shape[1], y.shape[1]]
        bias_shape = [y.shape[1]]
        self.W = np.random.random(weight_shape)
        self.b = np.random.random(bias_shape)
        for _ in range(num_iter):
            self.__gradientDescent(X, y)

    def predict(self, x):
        return np.dot(x, self.W) + self.b

    def __cost(self, X, y):
        yHat = self.predict(X)
        J = 0.5 * np.sum((yHat - y) ** 2)
        return J

    def __gradientDescent(self, X, y):
        m = len(X)
        w_gradient = 0
        b_gradient = 0
        for i, _ in enumerate(X):
            x = X[i]
            y_ = y[i]
            w_gradient += (self.predict(x) - y_) / m
            b_gradient += (self.predict(x) - y_) / m
        self.W = self.W - (self.learning_rate * w_gradient)
        self.b = self.b - (self.learning_rate * b_gradient)


if __name__ == '__main__':

    # !- Using the model
    clf = LinearRegression()
    clf.fit(X, y)
    print(clf.predict(X_pred))
    # !- Visualizing the model
    plt.scatter(X[:, 0], y)
    plt.plot(X_pred[:, 0], clf.predict(X_pred))
    plt.show()
