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
        return self.map_radius * math.exp(-iteration/self.t_const)
