"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 04 September, 2017 @ 9:58 PM.
  Copyright (c) 2017. Victor. All rights reserved.
"""

from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt


def visualize(X, centroids, labels):
    colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
    for i, x in enumerate(X):
        plt.scatter(x[0], x[1], s=50, c=colors[labels[i]])
    plt.scatter(centroids[:, 0], centroids[:, 1], s=70, marker='x', linewidth=3)
    plt.show()


X = np.array([
    [1, 2],
    [8, 7],
    [3, 9],
    [2, 6],
    [4, 1],
    [7, 9],
    [8, 8],
    [7, 7],
    [1, 7],
    [1, 8],
    [4, 2],
    [2.5, 3]
])

clf = KMeans(n_clusters=3)
clf.fit(X)
labels = clf.labels_
centroids = clf.cluster_centers_

visualize(X, centroids, labels)
# clf.visualize(X)
