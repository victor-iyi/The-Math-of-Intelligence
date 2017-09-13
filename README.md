# The-Math-of-Intelligence

## Implementation of Artificial Intelligence models without using any blackbox or libraries  😎

A high level API that implements common *Machine Learning algorithms* from scratch with only `numpy` as dependency.

> Available Machine Learning Algorithms include:

* Linear Regression
* Logistic Regression
* Support Vector Machine
* KMeans
* Self Organizing Map
* The Perceptron
* Multi-Layer Perceptron (n_layers)
* Convolutional Neural Network

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

> Requirements

Installing `numpy` using the Python Package Manager `pip`
```
pip install numpy
```

Installing `matplot` using the Python Package Manager `pip` to visualize data
```
pip install matplotlib
```

> Credits

[Siraj Raval](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A)
