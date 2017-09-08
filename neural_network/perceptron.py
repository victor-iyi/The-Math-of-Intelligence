"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 08 September, 2017 @ 6:03 PM.
  Copyright (c) 2017. Victor. All rights reserved.
"""

import numpy as np


class Perceptron(object):
    def __init__(self):
        self.W = None

    def fit(self, X, y):
        shape = X.shape
        self.W = np.random.rand(shape)

    def predict(self, X):
        return np.dot(X, self.W)

    def __sigmoid(self, X):
        return 1 / (1 + np.exp(-X))