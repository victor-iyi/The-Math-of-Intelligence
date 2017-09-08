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
        self.layers = layers if layers else []
        self.n_layers = len(layers)
        self.learning_rate = learning_rate

    def fit(self, X, y, n_iter=10000):
        num_classes = len(y[0])
        # !- Perform forward propagation
        y_hat = self.forward_prop(X, num_classes)

    def forward_prop(self, X, num_classes):
        hidden_layers = dict()
        for i, layer in enumerate(self.layers):
            prev = len(X[0]) if i == 0 else self.layers[i - 1]
            hidden_layers[i] = dict()
            hidden_layers[i]['weight'] = np.random.random([prev, layer])
            # hidden_layers[i]['bias'] = np.random.random([self.layers])
        output_layer = {'weight': np.random.random([self.layers[-1], num_classes])}
        activation = list()
        for n in range(self.n_layers):
            x = activation[-1] if n > 0 else X
            act = self.__sigmoid(np.dot(x, hidden_layers[n]['weight']))
            activation.append(act)
        y_hat = self.__sigmoid(np.dot(activation[-1], output_layer['weight']))
        return y_hat

    @staticmethod
    def __sigmoid(X, derivative=False):
        return 1 / (1 + np.exp(-X)) if not derivative else X * (1 - X)


if __name__ == '__main__':
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1]])
    y = np.array([[0], [0], [1], [1]])

    clf = MultiLayerPerceptron([5, 4, 3])
    clf.fit(X, y)

