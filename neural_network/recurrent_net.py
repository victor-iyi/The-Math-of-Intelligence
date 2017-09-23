"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 23 September, 2017 @ 2:29 PM.
  Copyright © 2017. Victor. All rights reserved.
"""

import numpy as np

book_name = r'../datasets/kafka.txt'

# Load in the data
data = open(book_name, 'r', encoding="utf-8").read()
chars = list(set(data))

# Data size and vocab size
data_size = len(data)
vocab_size = len(chars)
print('data_size = {:,}\t vocab_size = {:,}\n'.format(data_size, vocab_size))

# Char to index and index to char
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Vectorize a
"""
vector_a = np.zeros(shape=(vocab_size, 1))
a_idx = char_to_idx['a']
vector_a[a_idx] = 1

print('a is at index : {}'.format(a_idx))
print(vector_a.ravel())
"""
# Hyperparameters
hidden_size = 100
seq_len = 25
learning_rate = 1e-1

# Model parameters
# weights
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01   # input > hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden > hidden
Wyh = np.random.randn(vocab_size, hidden_size) * 0.01   # hidden > output
# biases
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))
