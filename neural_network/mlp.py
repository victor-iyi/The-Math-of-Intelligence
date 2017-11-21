"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 08 September, 2017 @ 9:17 PM.
  Copyright (c) 2017. Victor. All rights reserved.
"""
import sys
import numpy as np


class MultiLayerPerceptron(object):
    def __init__(self, layers=None, learning_rate=1e-3):
        self.layers = layers if layers else []
        self.learning_rate = learning_rate
        self.n_layers = len(self.layers)
        self.__weights = []
        self.__activation = []

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
            error, deltas = self.__train(X, y)
            error = np.mean(np.abs(error))
            self.__weight_update(deltas=deltas, X=X)
            sys.stdout.write(f'\rError: {error:.2f}')

    def inference(self, X):
        self.__activation = []
        for n in range(self.n_layers):
            x = X if n == 0 else self.__activation[-1]
            logit = np.dot(x, self.__weights[n])
            act = self.__sigmoid(logit)
            self.__activation.append(act)
        yHat = self.__sigmoid(np.dot(self.__activation[-1], self.__weights[-1]))
        return yHat

    def __train(self, X, y):
        yHat = self.inference(X)
        deltas = []
        error = None
        delta = 0
        for n in reversed(range(self.n_layers)):
            # if n == self.n_layers - 1:
            #     error = np.square(y - yHat)
            #     delta = error * self.__sigmoid(yHat, derivative=True)
            # else:
            #     print('Weights:', len(self.__weights), [w.T.shape for w in self.__weights])
            #     print(delta.shape, self.__weights[n+2].T.shape)
            #     error = np.dot(delta, self.__weights[n].T)
            #     delta = error * self.__sigmoid(self.__activation[n-1], derivative=True)
            print(n)
            error = np.square(y - yHat) if n == self.n_layers-1 else np.dot(delta, self.__weights[n+1].T)
            # delta = error * self.__sigmoid(self.__activation[n], derivative=True)
            delta = error * self.__sigmoid(yHat) if n == self.n_layers-1 else error * self.__sigmoid(self.__activation[
                                                                                                     n+1], derivative=True)
            deltas.append(delta)
        # print('Train deltas:', [d.shape for d in deltas])
        return error, deltas

    def __weight_update(self, deltas, X):
        deltas = list(reversed(deltas))
        print('Weights:', len(self.__weights), [w.shape for w in self.__weights])
        print('Activations:', len(self.__activation), [a.T.shape for a in self.__activation])
        print('deltas:', len(deltas), [d.shape for d in deltas])
        for i, _ in enumerate(self.__weights):
            layer = X if i == 0 else self.__activation[i-1]
            print(i, self.__weights[i].shape, layer.T.shape, deltas[i+1].shape)
            self.__weights[i] += np.dot(layer.T, deltas[i])

    @staticmethod
    def __sigmoid(X, derivative=False):
        return 1 / (1 + np.exp(-X)) if not derivative else X * (1 - X)


if __name__ == '__main__':
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1]])
    y = np.array([[0], [0], [1], [1]])

    clf = MultiLayerPerceptron(layers=[5, 4, 3, 3])
    clf.fit(X, y, n_iter=10000)
