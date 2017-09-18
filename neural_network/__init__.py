"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 08 September, 2017 @ 6:02 PM.
  Copyright (c) 2017. Victor. All rights reserved.
"""
from .perceptron import Perceptron
from .mlp import MultiLayerPerceptron

"""
The :mod:`linear_model` module implements generalized linear models. It
includes Linear Regression, Logistic Regression
"""

__all__ = [
    'Perceptron',
    'MultiLayerPerceptron',
]
