import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numba import jit


def load_test_data():
    data_list = [] 
    for i in range(5):
        file = "datasets/test_batch_{}".format(i)
        with open(file,'rb') as f:
            data_list.append(pickle.load(file = f))
    return pd.concat(data_list)

def load_train_data():
    data_list = [] 
    for i in range(3):
        file = "datasets/train_batch_{}".format(i)
        with open(file,'rb') as f:
            data_list.append(pickle.load(file = f))
    return pd.concat(data_list)

#deprecated
def split_train(data,prop,rso=1):
    y = data['target']
    data.drop(['id','target'], axis=1, inplace=True)
    return train_test_split(data,y ,train_size=prop,random_state=rso)

def abandon_col(data,rso=1):
    y = data['target']
    data.drop(['id','target'], axis=1, inplace=True)
    return data, y

@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def eval_gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]
