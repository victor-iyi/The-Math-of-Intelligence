"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 08 September, 2017 @ 6:03 PM.
  Copyright (c) 2017. Victor. All rights reserved.
"""

import numpy as np


class Perceptron(object):
    def __init__(self, learning_rate=1e-4):
        self.W = None
        self.learning_rate = learning_rate

    def fit(self, X, y, n_iter=10000):
        self.W = 2 * np.random.random((3, 1)) - 1
        for _ in range(n_iter):
            y_hat = self.predict(X)
            error = (y - y_hat) ** 2
            delta = np.multiply(error, self.__sigmoid(y_hat))
            gradient = np.dot(X.T, delta)
            self.W = self.W - (gradient * self.learning_rate)

    def predict(self, X):
        """
        Predict new inputs.

        :param X:
            Input to be predicted in form of `np.ndarray`
        :return:
        """
        return np.argmax(self.__sigmoid(np.dot(X, self.W)), axis=1)

    def score(self, X, y):
        y_hat = np.round(self.predict(X))
        correct = 0
        for i, y_ in enumerate(y):
            if y_hat[i] == y_:
                correct += 1
        return correct / len(y)

    @staticmethod
    def __sigmoid(X, derivative=False):
        return 1 / (1 + np.exp(-X)) if not derivative else np.exp(-X) / ((1 + np.exp(-X)) ** 2)


if __name__ == '__main__':
    # !- Dataset
    X_train = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y_train = np.array([[0], [0], [0], [1], [1]])
    X_test = np.array([[1, 0, 0], [1, 1, 0]])
    y_test = np.array([[1], [1]])

    # !- train and predict
    clf = Perceptron()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    # from sklearn.metrics import accuracy_score
    # accuracy = accuracy_score(y_test, np.round(y_pred))

    # !- Log the results
    from pprint import pprint

    pprint('y_test: {}'.format(y_test))
    pprint('y_pred: {}'.format(np.round(y_pred)))
    print('Accuracy = {:.02%}'.format(accuracy))
