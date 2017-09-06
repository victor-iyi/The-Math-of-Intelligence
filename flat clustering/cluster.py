"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 04 September, 2017 @ 9:55 PM.
  Copyright (c) 2017. Victor. All rights reserved.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean


class KMeans(object):
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-3):
        self.cluster_centers_ = dict()
        self.classes_ = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        # !- Select first k data points as centroids
        for i in range(self.n_clusters):
            self.cluster_centers_[i] = X[i]
        # !- Optimization process
        for _ in range(self.max_iter):
            self.classes_ = {}
            for i in range(self.n_clusters):
                self.classes_[i] = []
            # !- Loop through all features and calculate distance
            for x in X:
                label = self.predict(x)
                self.classes_[label].append(x)
            # print(self.classes_)
            prev_cluster_centers = dict(self.cluster_centers_)
            # !- Compute the mean of the classified data
            for label in self.classes_:
                self.cluster_centers_[label] = np.mean(self.classes_[label], axis=0)
            # !- Optimization
            optimized = False
            for label in self.cluster_centers_:
                old = prev_cluster_centers[label]
                new = self.cluster_centers_[label]
                percent_change = (new - old) / old * 100
                if np.sum(percent_change) < self.tol:
                    optimized = True
                    break
            if optimized:
                break

    def predict(self, X):
        dists = [self.__distance(self.cluster_centers_[centroid], X) for centroid in self.cluster_centers_]
        return dists.index(min(dists))

    @classmethod
    def __distance(cls, a, b):
        # return euclidean(a, b)
        return np.linalg.norm(a - b)

    @staticmethod
    def visualize(X, centroids=None, labels=None):
        colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
        if labels:
            for i, x in enumerate(X):
                plt.scatter(x[0], x[1], s=50, c=colors[labels[i]])
        else:
            plt.scatter(X[:, 0], X[:, 1], s=50)
        if centroids:
            for centroid in centroids:
                centroid = centroids[centroid]
                plt.scatter(centroid[0], centroid[1], s=70, c='k', marker='x', linewidth=3)
        plt.show()


if __name__ == '__main__':
    X = np.array([
        [1, 2], [8, 7], [3, 9], [2, 6], [4, 1], [7, 9],
        [8, 8], [7, 7], [1, 7], [1, 8], [4, 2], [2.5, 3]
    ])
    # X = np.array([[1, 2], [3, 2], [2, 3], [3, 1], [6, 8], [8, 6], [7, 7], [8, 9], [9, 7], [8, 8]])
    clf = KMeans(n_clusters=3)
    clf.fit(X)
    labels = [clf.predict(x) for x in X]
    clf.visualize(X, clf.cluster_centers_, labels)
