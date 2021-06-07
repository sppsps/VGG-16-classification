#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type = float, 
    default = 1e-2, help = 'learning rate')
parser.add_argument('--dropout', type = float, 
    default = 5e-1, help = 'dropout')
parser.add_argument('--regularizer', type = float, 
    default = 5e-4, help = 'regularizer')
parser.add_argument('--epoch', type = int, 
    default = 50, help = 'max number of epoch')
parser.add_argument('--batch_size', type = int, 
    default = 64, help = '# of batch size')
parser.add_argument('--momentum', type = float, 
    default = 0.9, help = 'momentum')
parser.add_argument('--patience', type = int, 
    default = 4, help = 'patience')
# import easydict
# args = easydict.EasyDict({
#         "learning_rate": 0.01,
#         "batch_size": 64,
#         "epoch": 50,
#         "momentum": 0.9,
#         "patience": 4,
#         "dropout": 0.5,
#         "regularizer": 0.0005
# })
def training(args):
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.val_losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
    history1 = LossHistory()
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=0, mode='min', patience = args.patience)
    mcp_save = keras.callbacks.ModelCheckpoint('vgg16_1.h5', save_best_only=True, monitor='val_acc', mode='max')
    my_model = vgg_model(args.learning_rate, args.momentum, args.regularizer, args.dropout)
    history = my_model.fit(
        x_train,y_train,batch_size = args.batch_size, epochs = args.epoch, validation_data = (x_test,y_test), callbacks = [mcp_save,earlyStopping]
    )
args, unknown = parser.parse_known_args()
training(args)
