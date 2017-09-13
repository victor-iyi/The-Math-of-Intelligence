"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 13 September, 2017 @ 8:04 PM.
  Copyright © 2017. Victor. All rights reserved.
"""
import numpy as np
import math


class SelfOrganizingMap(object):
    def __init__(self, X_size, y_size, trait_num, t_iter, t_step):
        self.weights = np.random.randint(256, size=(X_size, y_size, trait_num)).astype('float64')
        self.t_iter = t_iter
        self.map_radius = max(self.weights.shape) / 2
        self.t_const = self.t_iter / math.log(self.map_radius)
        self.t_step = t_step

    def __distanceMatrix(self, vector):
        return np.sum((self.weights - vector) ** 2, 2)

    def __bestMatchingUnit(self, vector):
        distance = self.__distanceMatrix(vector)
        return np.unravel_index(distance.argmin(), distance.shape)

    def __bumDistance(self, vector):
        x, y, rgb = self.weights.shape
        xi = np.arange(x).reshape(x, 1).repeat(y, 1)
        yi = np.arange(y).reshape(1, y).repeat(x, 0)
        return np.sum((np.dstack((xi, yi)) - np.array(self.__bestMatchingUnit(vector))) ** 2, 2)

    def __hoodRadius(self, iteration):
        return self.map_radius * math.exp(-iteration / self.t_const)

    def __trainRow(self, vector, i, dist_cut, dist):
        hood_radius_2 = self.__hoodRadius(i) ** 2
        bum_distance = self.__bumDistance(vector).astype('float64')
        if dist is None:
            temp = hood_radius_2 - bum_distance
        else:
            temp = dist ** 2 - bum_distance
        influence = np.exp(-bum_distance / (2 * hood_radius_2))
        if dist_cut:
            influence *= ((np.sign(temp) + 1) / 2)
        return np.expand_dims(influence, 2) * (vector - self.weights)

    def fit(self, t_set, distance_cutoff=False, distance=None):
        for i in range(self.t_iter):
            for x in t_set:
                self.weights += self.__trainRow(x, i, distance_cutoff, distance)


if __name__ == '__main__':
    t_set = np.random.randint(256, size=(15, 3))  # generate data
    som = SelfOrganizingMap(200, 200, 3, 100, 0.1)  # init model
    som.fit(t_set=t_set)
