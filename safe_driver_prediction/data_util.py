import pickle
import pandas as pd

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
