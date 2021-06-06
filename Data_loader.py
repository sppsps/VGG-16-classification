#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)
x_train = x_train.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)
x_train = x_train/255.0
x_test = x_test/255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

