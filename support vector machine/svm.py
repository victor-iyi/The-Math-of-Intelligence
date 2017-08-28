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
    def __init__(self):
        self.W = 0

    def fit(self, X, y):
        pass

    def predict(self):
        pass

    def __sgd(self):
        pass

    @classmethod
    def plot(cls, X, line_X, line_y):
        for i, sample in enumerate(X):
            if i < 2:
                plt.scatter(sample[0], sample[1], s=100, c='red', marker='_', linewidths=3)
            else:
                plt.scatter(sample[0], sample[1], s=100, c='blue', marker='+', linewidths=3)
        plt.plot(line_X, line_y, color='k', linewidth=2)
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

    svm = SVC()
    svm.fit(X, y)
    svm.plot(X, [-2, 6], [6, 0.5])
