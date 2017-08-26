"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 26 August, 2017 @ 10:20 PM.
  Copyright (c) 2017. victor. All rights reserved.
"""

import numpy as np


class SVC(object):
    def __init__(self):
        self.w = 0


def fit(self, X, y):
    pass


def predict(self):
    pass


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
