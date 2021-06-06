#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
history1 = LossHistory()
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=0, mode='min', patience = 4)
mcp_save = keras.callbacks.ModelCheckpoint('vgg16_1.h5', save_best_only=True, monitor='val_acc', mode='max')
history = my_model.fit(
    x_train,y_train,batch_size = 64, epochs = 50, validation_data = (x_test,y_test), callbacks = [mcp_save,earlyStopping]
)

