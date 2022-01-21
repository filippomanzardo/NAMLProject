import pandas as pd 
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from preprocess_data import *
from train_test_models import *
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix

def experiment(data, num_iteration):

    train_x, train_y, test_x, test_y = data
    learning_rates = [0.1, 0.001, 0.0001, 0.01, 0.5, 0.05, 0.005, 0.0005] 
    batch_sizes = [64, 128, 256, 512, 1024]
    epochs = [20, 30, 40, 50] 

    for i in range(num_iteration):
    
        print(f"\n---------------------- ITERATION {i+1} ----------------------\n")
    
        lr = np.random.choice(learning_rates)
        bs = np.random.choice(batch_sizes)
        ep = np.random.choice(epochs)

        print(f"Learning Rate: {lr}, Batch Size: {bs}, Epochs: {ep}\n")

        num_nets = 3

        models = list()
        train_accuracies = list()

        for i in range(num_nets):
            net, history = train_model(net=i, train_data=(train_x, train_y), 
                                            batch_size=bs,
                                            learning_rate=lr, 
                                            epochs=ep,  
                                            loss='binary_crossentropy',
                                            verbose=0, return_history=True)
            models.append(net)
            train_accuracies.append(history.history['accuracy'][0])

        accuracies = list()
        for i in range(num_nets):
            accuracies.append(test_model(models[i], test_x, test_y, only_accuracy=True))

        print(f"\n----------- Train -----------")
        print(f"Accuracy FullyConnected Network: {train_accuracies[0]:.4f}")
        print(f"Accuracy LSTM Network: {train_accuracies[1]:.4f}")
        print(f"Accuracy CNN Network: {train_accuracies[2]:.4f}\n")

        print(f"\n----------- Test -----------")
        print(f"Accuracy FullyConnected Network: {accuracies[0]:.4f}")
        print(f"Accuracy LSTM Network: {accuracies[1]:.4f}")
        print(f"Accuracy CNN Network: {accuracies[2]:.4f}\n")

        features = list()
        features_x = list()
        for i in range(num_nets):
            features.append(keras.Model(
                inputs=models[i].inputs, 
                outputs=models[i].layers[-4].output
            ))
            features_x.append(features[i](train_x).numpy())

        features_conc = np.concatenate((features_x), axis=1)


        param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear','sigmoid']}
        classifier = GridSearchCV(svm.SVC(class_weight='balanced'), param_grid)
        classifier = classifier.fit(features_conc, train_y[:,0])

        test_features = list()

        for i in range(num_nets):
            test_features.append(features[i](test_x).numpy())

        test_con = np.concatenate((test_features), axis =  1)

        y_hat = classifier.predict(test_con)

        print(f"\n----------- SVM -----------")

        print("Accuracy: {:.4f}\n".format(np.count_nonzero(((y_hat == test_y[:,0])))/len(test_y)))


if __name__ == "__main__":
    
    print("Starting...")

    timeframe = "6h"
    data = pd.read_csv(f"../data/BTC_EUR-{timeframe}.csv")
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms')
    data.set_index(keys='Timestamp', inplace=True)

    shift_days = 10
    window = 24
    value_to_predict = 'Close'

    print("Preprocessing data...")
    train_x, val_x, test_x, train_y, val_y, test_y = preprocess_data(data, k=shift_days, column=value_to_predict, window=window)

    
    num_iterations = 3
    data = (train_x, train_y, test_x, test_y)

    print("Iteration starting...")
    experiment(data, num_iterations)