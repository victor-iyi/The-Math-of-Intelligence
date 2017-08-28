"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 26 August, 2017 @ 10:20 PM.
  Copyright (c) 2017. victor. All rights reserved.
"""

import numpy as np
from matplotlib import pyplot as plt


class SVC(object):
    def __init__(self, learning_rate=1):
        self.W = None
        self.learning_rate = learning_rate
        self.__lambda = None

    def fit(self, X, y, epochs=10000):
        self.__lambda = 1 / epochs
        self.W = np.zeros(len(X[0]))
        # store misclassified points
        errors = []
        print('Training starting...')
        for epoch in range(1, epochs + 1):
            error = 0
            for i, x in enumerate(X):
                # noinspection PyTypeChecker
                if (y[i] * np.dot(X[i], self.W)) < 1:
                    # Misclassified points
                    self.W = self.W + self.learning_rate * (X[i] * y[i] + (-2 * self.__lambda * self.W))
                    error = 1
                else:
                    # Correct classification
                    self.W = self.W + self.learning_rate * (-2 * self.__lambda * self.W)
            errors.append(error)
        self.__plot_error(errors)

    def predict(self, X):
        return np.dot(X, self.W)

    @classmethod
    def plot(cls, X, line_X, line_y):
        for i, sample in enumerate(X):
            if i < 2:
                plt.scatter(sample[0], sample[1], s=100, c='red', marker='_', linewidths=3)
            else:
                plt.scatter(sample[0], sample[1], s=100, c='blue', marker='+', linewidths=3)
        plt.plot(line_X, line_y, color='k', linewidth=2)
        plt.show()

    @staticmethod
    def __plot_error(errors):
        plt.plot(errors, '|')
        plt.ylim(0.5, 1.5)
        plt.axes().set_yticklabels([])
        plt.xlabel('Epochs')


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

    svm = SVC()
    svm.fit(X, y)
    svm.plot(X, [-2, 6], [6, 0.5])
