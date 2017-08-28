"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 26 August, 2017 @ 10:33 PM.
  Copyright (c) 2017. victor. All rights reserved.
"""

import numpy as np
from svm import SupportVectorMachine

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

svm = SupportVectorMachine()
svm.fit(X, y, show_metric=True)
svm.plot(X, show_hyperplane=True)
