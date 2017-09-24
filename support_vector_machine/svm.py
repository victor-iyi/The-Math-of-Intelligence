"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 26 August, 2017 @ 10:20 PM.
  Copyright © 2017. victor. All rights reserved.
"""

import numpy as np
from matplotlib import pyplot as plt


class SupportVectorMachine(object):
    def __init__(self, learning_rate=1):
        self.W = None
        self.learning_rate = learning_rate
        self.__lambda = None

    def fit(self, X, y, epochs=100000, **kwargs):
        self.W = np.zeros(len(X[0]))
        # store misclassified points
        errors = []
        print('Training starting...')
        for epoch in range(1, epochs + 1):
            error = 0
            self.__lambda = 1 / epoch
            for i, x in enumerate(X):
                regularizer = -2 * self.__lambda * self.W
                # noinspection PyTypeChecker
                if (y[i] * np.dot(X[i], self.W)) < 1:
                    # Misclassified points
                    self.W = self.W + self.learning_rate * (X[i] * y[i] + regularizer)
                    error = 1
                else:
                    # Correct classification
                    self.W = self.W + self.learning_rate * regularizer
            errors.append(error)
        # Show errors reducing over time
        if 'show_metric' in kwargs:
            if kwargs['show_metric']:
                self.__plot_error(errors)

    def predict(self, X):
        return np.dot(X, self.W)

    def plot(self, X, **kwargs):
        for i, sample in enumerate(X):
            if i < 2:
                plt.scatter(sample[0], sample[1], s=100, c='r', marker='_', linewidths=3)
            else:
                plt.scatter(sample[0], sample[1], s=100, c='b', marker='+', linewidths=3)
        if 'show_hyperplane' in kwargs:
            if kwargs['show_hyperplane']:
                x2 = [self.W[0], self.W[1], -self.W[1], self.W[0]]
                x3 = [self.W[0], self.W[1], self.W[1], -self.W[0]]
                x2x3 = np.array([x2, x3])
                x, y, u, v = zip(*x2x3)
                ax = plt.gca()
                ax.quiver(x, y, u, v, scale=1, color='k')
        plt.show()

    @staticmethod
    def __plot_error(errors):
        plt.plot(errors, '|')
        plt.ylim(0.5, 1.5)
        plt.axes().set_yticklabels([])
        plt.xlabel('Epochs')
        plt.ylabel('Misclassified')
        plt.show()


if __name__ == '__main__':
    # Input [x, y, bias]
    X = np.array([
        [-2, 4, -1],
        [4, 1, -1],
        [1, 6, -1],
        [2, 4, -1],
        [6, 2, -1]
    ])
    # Labels
    y = np.array([-1, -1, 1, 1, 1])

    # Input [x, y, bias]
    # X = np.array([
    #     [0, 0, 1],
    #     [0, 1, 0],
    #     [0, 1, 1],
    #     [1, 0, 1],
    #     [1, 1, 1],
    # ])
    # Labels
    # y = np.array([[0], [0], [0], [1], [1]])

    # Support Vector Classifier
    clf = SupportVectorMachine()
    clf.fit(X, y, show_metric=True)
    X_pred = np.array([
        [8, 2, -1],
        [4, 1, -1],
        [6, 7, -1],
    ])
    # X_pred = np.array([
    #     [1, 1, 0],
    #     [1, 0, 0],
    # ])
    pred = clf.predict(X_pred)
    print('Predictions:', pred)
    clf.plot(X, show_hyperplane=True)
