import tensorflow as tf 
import numpy as np 
import sys 

sys.path.insert(0, '../models/')

from sklearn.metrics import confusion_matrix
from tensorflow import keras 
from keras import layers

def test_model(model, test_x, test_y, only_accuracy=False):
    y_hat = model.predict(test_x)
    for i in range(y_hat.shape[0]):
        if y_hat[i][0] < 0.5:
            y_hat[i][0] = 0.
        else:
            y_hat[i][0] = 1. 

    buys = test_y.sum()
    holds = len(test_y)-test_y.sum()

    cm = confusion_matrix(test_y, y_hat)
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    accuracy = np.count_nonzero((y_hat == np.array(test_y)))/len(test_y)

    if not only_accuracy:
        print('Buys: {:d}, Holds: {:d}'.format(buys, holds))
        print("True Buys: {:d}, False Buys: {:d}, True Holds: {:d}, False Holds: {:d}".format(TP[0],FP[0],TN[0],FN[0]))
    print("Accuracy: {:.4f}".format(accuracy))

    return accuracy

def testmodel(input_shape, activation='relu'):

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(units=256, activation=activation))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=1, activation='sigmoid'))

    return model

def model0(input_shape, activation='relu', optimizer=None, loss=None):

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape, name='input_layer'))
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(units=512, activation=activation, name='dense_1'))
    model.add(layers.Dense(units=256, activation=activation, name='dense_2'))
    model.add(layers.Dense(units=128, activation=activation, name='dense_3'))
    model.add(layers.Dense(units=64, activation=activation, name='dense_4'))
    model.add(layers.Dense(units=256, activation=activation, name='features_output'))
    model.add(layers.Dense(units=32, activation=activation, name='dense_5'))
    model.add(layers.Dense(units=16, activation=activation, name='dense_6'))
    model.add(layers.Dense(units=1, activation='sigmoid', name='output'))

    return model

def model1(input_shape, activation='relu'):

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape, name='input_layer'))

    model.add(layers.LSTM(units=256, return_sequences=True, name='LSTM_1'))
    model.add(layers.Dropout(0.4, name='dropout_1'))
    model.add(layers.BatchNormalization(name='batchnorm_1'))

    model.add(layers.LSTM(units=128, return_sequences=False, name='LSTM_2'))
    model.add(layers.Dropout(0.2, name='dropout_2'))
    model.add(layers.BatchNormalization(name='batchnorm_2'))

    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(units=256, activation=activation, name='features_output'))
    model.add(layers.Dense(units=64, activation=activation, name='dense_1'))
    model.add(layers.Dense(units=32, activation=activation, name='dense_2'))
    model.add(layers.Dense(units=1, activation='sigmoid', name='output'))

    return model

def model2(input_shape, activation='relu'):

    model = keras.Sequential()
    model.add(layers.Input(input_shape, name='input_layer'))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation=activation, name='convolution1D_1'))
    model.add(layers.Dropout(0.5, name='dropout_1'))
    model.add(layers.AveragePooling1D(pool_size=2, name='average_pooling_1'))

    model.add(layers.Conv1D(filters=32, kernel_size=3, activation=activation, name='convolution1D_2'))
    model.add(layers.Dropout(0.2, name='dropout_2'))
    model.add(layers.AveragePooling1D(pool_size=2, name='average_pooling_2'))

    model.add(layers.Conv1D(filters=16, kernel_size=1, activation=activation, name='convolution1D_3'))
    model.add(layers.Dropout(0.2, name='dropout_3'))
    model.add(layers.GlobalAveragePooling1D(name='global_average'))

    model.add(layers.Dense(units=256, activation=activation, name='features_output'))
    model.add(layers.Dense(units=32, activation=activation, name='dense_1'))
    model.add(layers.Dense(32, activation=activation, name='dense_2'))
    model.add(layers.Dense(1, activation='sigmoid', name='output'))
    
    return model

def train_model(net, train_data, val_data=None, batch_size=32, 
                epochs=100, activation=None, loss=None,
                verbose=1, learning_rate=0.001, return_history=False):

    train_x, train_y = train_data
    input_shape = train_x.shape[1:]


    if net == 0:
        model = model0(input_shape, activation)
    elif net == 1:
        model = model1(input_shape, activation)
    elif net == 2:
        model = model2(input_shape, activation)
    elif net == -1:
        model = testmodel(input_shape, activation)
    

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = 'binary_crossentropy' if loss == None else loss

    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    
    history = model.fit(train_x, 
            train_y, 
            verbose=verbose, 
            batch_size=batch_size, 
            epochs=epochs,
            validation_data=val_data,
    )

    if (return_history):
        return model, history

    return model