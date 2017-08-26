"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 26 August, 2017 @ 10:33 PM.
  Copyright (c) 2017. victor. All rights reserved.
"""

import numpy as np
from matplotlib import pyplot as plt

# Define the data
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

for d, sample in enumerate(X):
    if d < 2:
        plt.scatter(sample[0], sample[1], marker='_', linewidths=5)
    else:
        plt.scatter(sample[0], sample[1], marker='+', linewidths=5)

plt.plot([-2, 6], [6, 0.5])  # A toy decision boundary
plt.show()
