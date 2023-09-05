# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 12:52:35 2022

@author: turnerp
"""


import numpy as np
import pandas as pd
import matplotlib
import random


import matplotlib.pyplot as plt





pattern_len=[0.1]
pattern_pos=[0.1, 0.4,0.8]
ts_len=128
ts_n=3
    
random.seed(1234)
np.random.seed(1234)

nb_classes = len(pattern_pos) * len(pattern_len)

x_train = np.random.normal(0.0, 0.1, size=(ts_n, ts_len))
x_test = np.random.normal(0.0, 0.1, size=(ts_n, ts_len))

y_train = np.random.randint(low=0, high=nb_classes, size=(ts_n,))
y_test = np.random.randint(low=0, high=nb_classes, size=(ts_n,))

# make sure at least each class has one example
y_train[:nb_classes] = np.arange(start=0, stop=nb_classes, dtype=np.int32)
y_test[:nb_classes] = np.arange(start=0, stop=nb_classes, dtype=np.int32)

# each class is defined with a certain combination of pattern_pos and pattern_len
# with one pattern_len and two pattern_pos we can create only two classes
# example:  class 0 _____-_  & class 1 _-_____

# create the class definitions
class_def = [None for i in range(nb_classes)]

idx_class = 0
for pl in pattern_len:
    for pp in pattern_pos:
        class_def[idx_class] = {'pattern_len': int(pl * ts_len),
                                'pattern_pos': int(pp * ts_len)}
        idx_class += 1

# # create the dataset
for i in range(ts_n):
    # for the train
    c = y_train[i]
    curr_pattern_pos = class_def[c]['pattern_pos']
    curr_pattern_len = class_def[c]['pattern_len']
    x_train[i][curr_pattern_pos:curr_pattern_pos + curr_pattern_len] = \
        x_train[i][curr_pattern_pos:curr_pattern_pos + curr_pattern_len] + 1.0
        
    plt.plot(x_train[i])

#     # for the test
#     c = y_test[i]
#     curr_pattern_pos = class_def[c]['pattern_pos']
#     curr_pattern_len = class_def[c]['pattern_len']
#     x_test[i][curr_pattern_pos:curr_pattern_pos + curr_pattern_len] = \
#         x_test[i][curr_pattern_pos:curr_pattern_pos + curr_pattern_len] + 1.0

# # znorm
# x_train = (x_train - x_train.mean(axis=1, keepdims=True)) \
#           / x_train.std(axis=1, keepdims=True)

# x_test = (x_test - x_test.mean(axis=1, keepdims=True)) \
#          / x_test.std(axis=1, keepdims=True)
