"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 04 September, 2017 @ 9:58 PM.
  Copyright (c) 2017. Victor. All rights reserved.
"""

from .cluster import KMeans
from .self_organizing_map import SelfOrganizingMap

"""
The :mod:`cluster` module implements generalized clustering algorithms. It
includes KMeans, SelfOrganizingMap
"""

__all__ = [
    'KMeans',
    'SelfOrganizingMap'
]
