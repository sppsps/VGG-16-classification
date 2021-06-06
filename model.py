#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
def vgg_model():
  model = keras.Sequential()
  model.add(keras.Input(shape = (32,32,3)))
  model.add(layers.Conv2D(filters =64, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.Conv2D(filters =64, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))
  model.add(layers.Conv2D(filters =128, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.Conv2D(filters =128, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))
  model.add(layers.Conv2D(filters =256, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.Conv2D(filters =256, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.Conv2D(filters =256, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.Conv2D(filters =256, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))
  model.add(layers.Conv2D(filters =512, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.Conv2D(filters =512, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.Conv2D(filters =512, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.Conv2D(filters =512, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))
  model.add(layers.Conv2D(filters =512, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.Conv2D(filters =512, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.Conv2D(filters =512, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.Conv2D(filters =512, kernel_size=(3,3), strides = (1,1), padding = 'same', activation = 'relu'))
  model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))
  model.add(layers.Flatten())
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(4096, 'relu', True, kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
  model.add(layers.Dropout(0.5))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(4096, 'relu', True, kernel_regularizer=tf.keras.regularizers.l2(0.0005),))
  model.add(layers.Dropout(0.5))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(1000, 'relu', True, kernel_regularizer=tf.keras.regularizers.l2(0.0005))) 
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(10))
  model.add(layers.Activation('softmax'))
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False, name="SGD",)
  my_model.compile(
  optimizer = optimizer,
  loss = 'categorical_crossentropy',
  metrics = ['accuracy']
  )
  return model

