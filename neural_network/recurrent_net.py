"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 23 September, 2017 @ 2:29 PM.
  Copyright © 2017. Victor. All rights reserved.
"""

book_name = r'../datasets/kafka.txt'

data = open(book_name, 'r', encoding="utf-8").read()
chars = list(set(data))

# Data size and vocab size
data_size = len(data)
vocab_size = len(chars)

print('data_size = {:,}\t vocab_size = {:,}'.format(data_size, vocab_size))
