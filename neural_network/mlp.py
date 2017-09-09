"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 08 September, 2017 @ 9:17 PM.
  Copyright (c) 2017. Victor. All rights reserved.
"""

import numpy as np


class MultiLayerPerceptron(object):
    def __init__(self, layers=None, learning_rate=1e-3):
        self.__weights = []
        self.__activation = []
        self.n_layers = len(layers)
        self.layers = layers if layers else []
        self.learning_rate = learning_rate

    def fit(self, X, y, n_iter=10000):
        # !- Randomly initialize weights
        for i, layer in enumerate(self.layers):
            prev = len(X[0]) if i == 0 else self.layers[i - 1]
            weight = 2 * np.random.random([prev, layer]) - 1
            self.__weights.append(weight)
        # !- weights connecting to output layer
        self.__weights.append(2 * np.random.random([self.layers[-1], len(y[0])]) - 1)
        # !- Perform forward propagation
        for _ in range(n_iter):
            self.backwardProp(X, y)

    def forwardProp(self, X):
        self.__activation = []
        for n in range(self.n_layers):
            x = self.__activation[-1] if n > 0 else X
            logit = np.dot(x, self.__weights[n])
            act = self.__sigmoid(logit)
            self.__activation.append(act)
        yHat = self.__sigmoid(np.dot(self.__activation[-1], self.__weights[-1]))
        return yHat

    def backwardProp(self, X, y):
        yHat = self.forwardProp(X)
        error = (y - yHat) ** 2
        # TODO: Continue with back prop
        print(error)

    @staticmethod
    def __sigmoid(X, derivative=False):
        return 1 / (1 + np.exp(-X)) if not derivative else X * (1 - X)


if __name__ == '__main__':
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1]])
    y = np.array([[0], [0], [1], [1]])

    clf = MultiLayerPerceptron([5, 4, 3, 3])
    clf.fit(X, y)
