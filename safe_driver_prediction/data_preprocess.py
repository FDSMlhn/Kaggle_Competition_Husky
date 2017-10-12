from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd


class naive_preprocess():
    def __init__(self):
        self.scaler= StandardScaler()
        self.mean = None

    def dtype_change(self,data):
        data[data == -1] = None
        for col in data.columns:
            if 'bin' in col or 'cat' in col:
                data[col] = data[col].astype('category')
        data_1 = pd.get_dummies(data)
        return data_1
    
    def scale(self,data,test=None):
        if test is None:
            self.mean = data.mean()
            data = data.fillna(self.mean)
            self.scaler.fit(data)
        if test is True:
            data = data.fillna(self.mean)
        self.scaler.transform(data)
        return data
        

# def naive_preprocess():
#     scaler = StandardScaler()  
#     scaler.fit(train_data)  
#     train_data = scaler.transform(train_data,)  


#     train_data= train_data.iloc[:10000,:]
#     train_data = pd.get_dummies(train_data)
#     train_data.fillna(train_data.mean(),inplace=True)
#     dev_y = train_data['target']
#     train_data.drop(['id','target'], axis=1, inplace=True)