# The-Math-of-Intelligence

## Implementation of Artificial Intelligence models without using any blackbox or libraries  😎

A high level API that implements common *Machine Learning algorithms* from scratch with only `numpy` as dependency.

> Available Machine Learning Algorithms include:

+ Linear Regression
+ Logistic Regression
+ Support Vector Machine
+ KMeans
+ The Perceptron
+ Multi-Layer Perceptron (n_layers)
+ Convolutional Neural Network

> How to use
```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1]])
y = np.array([[0], [0], [1], [1]])

clf = LinearRegression()
clf.fit(X, y)
y_pred = clf.predict(np.array([[1, 1, 0]]))

print('Prediction: {}'.format(y_pred))

```
