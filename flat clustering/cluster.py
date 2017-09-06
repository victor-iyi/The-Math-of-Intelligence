"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 04 September, 2017 @ 9:55 PM.
  Copyright (c) 2017. Victor. All rights reserved.
"""
from matplotlib import pyplot as plt

class KMeans(object):
    def __init__(self, n_clusters=8, max_iter=300, tol=1e3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        pass

    def predict(self, X):
        pass

    @staticmethod
    def visualize(X):
        plt.scatter(X[:, 0], X[:, 1], s=50)
        plt.show()
