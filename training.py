import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from Data_loader import data_loader
from model import vgg_model
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
def training(args):
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.val_losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
    history_loss = LossHistory()
    x_train, y_train, x_test, y_test = data_loader()
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=0, mode='min', patience = args.patience)
    mcp_save = keras.callbacks.ModelCheckpoint('vgg16_1.h5', save_best_only=True, monitor='val_acc', mode='max')
    my_model = vgg_model(args.learning_rate, args.momentum, args.regularizer, args.dropout)
    history = my_model.fit(
        x_train,y_train,batch_size = args.batch_size, epochs = args.epoch, validation_data = (x_test,y_test), callbacks = [mcp_save,earlyStopping]
    )
    return history, history_loss
args, unknown = parser.parse_known_args()
history, history_loss  = training(args)
def plot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test','loss'], loc='upper left')
    plt.show()
plot(history)
