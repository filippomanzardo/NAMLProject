import pandas_ta as ta 
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque 
from random import shuffle 


def add_indicators(data):
    
    data_noLH=data.copy()

    data.ta.log_return(cumulative=True, append=True)
    data.ta.percent_return(cumulative=True, append=True)


    column_LH = ['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26', 'DPO_20']
    base_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    data.ta.strategy(ta.AllStrategy)
    data.dropna(axis=1, thresh=round(len(data)*0.9), inplace=True)
    
    for column in column_LH:
        try:
            data.drop(labels=column, inplace=True, axis=1)
        except:
            continue

    data_noLH.ta.dpo(lookhead=False)
    data_noLH.ta.ichimoku(lookhead=False)
    data_noLH.drop(labels=base_columns, inplace=True, axis=1)

    data = pd.concat([data, data_noLH], axis=1)
    data.dropna(axis=1, thresh=round(len(data)*0.9), inplace=True)
    data.dropna(axis=0, inplace=True)

    return data


def preprocess_data(data, k=1, column=None, window=None, split=0.8, 
                    validation_split=0, indicators=True, equalize=True):

    if(indicators):
        data = add_indicators(data)

    n = len(data)
    buy = np.zeros(n, dtype=np.int8)
    for i in range(k,n):
        if(data.iloc[i-k][column] < data.iloc[i][column]):
            buy[i-k] = np.int8(1)

    data['Buy'] = buy

    window_data = []
    labels = []
    prev_days_features= deque(maxlen=window)

    data = data.to_numpy()
    n = len(data)
    train_x = []
    val_x = []
    test_x = []
    num_train = int(split*n)
    num_validation = int(n*validation_split)
    
    train_x = data[0:num_train]
    val_x = data[num_train:num_train + num_validation] if(validation_split) else np.empty(shape=(0,0))
    test_x = data[num_train + num_validation:]

    # Faster if done by columns
    for i in range(train_x.shape[1]-1) :
        X = train_x[:, i].reshape(-1,1)
        scaler = StandardScaler().fit(X)
        train_x[:, i] = scaler.transform(X).reshape(1,-1)

    for i in range(val_x.shape[1]-1):
        X = val_x[:, i].reshape(-1,1)
        scaler = StandardScaler().fit(X)
        val_x[:, i] = scaler.transform(X).reshape(1,-1)

    for i in range(test_x.shape[1]-1):
        X = test_x[:, i].reshape(-1,1)
        scaler = StandardScaler().fit(X)
        test_x[:, i] = scaler.transform(X).reshape(1,-1)


    for i in train_x:  # iterate over the values
        prev_days_features.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days_features) == window:  
            window_data.append(np.array(prev_days_features))
            labels.append(np.int8(i[-1]))

    for i in val_x:  # iterate over the values
        prev_days_features.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days_features) == window:  
            window_data.append(np.array(prev_days_features))
            labels.append(np.int8(i[-1]))

    for i in test_x:  # iterate over the values
        prev_days_features.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days_features) == window:  
            window_data.append(np.array(prev_days_features))
            labels.append(np.int8(i[-1]))

    train_x = np.array(window_data[0:num_train])
    val_x = np.array(window_data[num_train:num_train + num_validation])
    test_x = np.array(window_data[num_train + num_validation:])

    train_y = np.array(labels[0:num_train]).reshape(-1,1)
    val_y = np.array(labels[num_train:num_train + num_validation]).reshape(-1,1)
    test_y = np.array(labels[num_train + num_validation:]).reshape(-1,1)

    if(equalize):
        n = len(train_x)
        n_buys = train_y.sum()
        n_holds = n - n_buys

        if n_buys > n_holds:
            n_delete = n_buys - n_holds
            j = 1
        else:
            n_delete = n_holds - n_buys
            j = 0

        shuffled = list(zip(train_x, train_y))
        shuffle(shuffled)

        (train_x, train_y) = zip(*shuffled)

        train_x, train_y = zip(*shuffled)
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y).reshape(-1,1)

        while(n_delete > 0):
            for i in range(len(train_x)):
                if(train_y[i] == j):
                    train_x = np.delete(train_x, i, axis = 0)
                    train_y = np.delete(train_y, i, axis = 0)
                    n_delete -= 1
                    break

        

    return train_x, val_x, test_x, train_y, val_y, test_y