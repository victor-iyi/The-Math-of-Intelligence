"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 26 August, 2017 @ 9:33 PM.
  Copyright © 2017. victor. All rights reserved.
"""

# Create a LinearRegression object
from regression import LinearRegression
import numpy as np

data = np.genfromtxt('data.csv', delimiter=',')
num_iter = 1000

clf = LinearRegression(learning_rate=1e-4)
clf.fit(data=data, num_iter=num_iter)
print('After {:,} iterations. m = {:.2f} and b = {:.2f}'.format(num_iter, clf.m, clf.b))
