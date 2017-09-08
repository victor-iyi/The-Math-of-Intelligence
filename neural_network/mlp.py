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
        self.hidden_layers = dict()
        self.output_layer = dict()
        self.activation = []
        self.n_layers = len(layers)
        self.layers = layers if layers else []
        self.learning_rate = learning_rate

    def fit(self, X, y, n_iter=10000):
        # !- Randomly initialize weights
        for i, layer in enumerate(self.layers):
            prev = len(X[0]) if i == 0 else self.layers[i - 1]
            self.hidden_layers[i] = dict()
            self.hidden_layers[i]['weight'] = 2 * np.random.random([prev, layer]) - 1
        self.output_layer['weight'] = 2 * np.random.random([self.layers[-1], len(y[0])]) - 1
        # !- Perform forward propagation
        y_hat = self.forward_prop(X)
        print(y_hat)

    def forward_prop(self, X):

        self.activation = []
        for n in range(self.n_layers):
            x = self.activation[-1] if n > 0 else X
            act = self.__sigmoid(np.dot(x, self.hidden_layers[n]['weight']))
            self.activation.append(act)
        y_hat = self.__sigmoid(np.dot(self.activation[-1], self.output_layer['weight']))
        return y_hat

    def backward_prop(self, X, y):
        y_hat = self.forward_prop(X)
        error = (y - y_hat) ** 2
        for i in range(self.n_layers):
            delta = np.multiply(error, self.__sigmoid(self.activation[i]))
            gradient = np.dot(delta, self.hidden_layers[i]['weight'].T)
            # !- Weight update
            self.hidden_layers[i]['weight'] -= (gradient * self.learning_rate)
            error = np.multiply(delta, self.__sigmoid(self.activation[i+1]))  # error for next layer

    @staticmethod
    def __sigmoid(X, derivative=False):
        return 1 / (1 + np.exp(-X)) if not derivative else X * (1 - X)


if __name__ == '__main__':
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1]])
    y = np.array([[0], [0], [1], [1]])

    clf = MultiLayerPerceptron([5, 4, 3])
    clf.fit(X, y)
