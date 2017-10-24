from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

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
            self.scaler = self.scaler.fit(data)
        if test is True:
            data = data.fillna(self.mean)
        data = self.scaler.transform(data)
        return data
        

class preprocess_cell():
    #General rules for preprocess
    #choose useful predictors
    #deal with NA it(mean or median), treat val, test data with same mean.
    #create dummy variable for cat
    #split train, val, test data (not included in this cell)
    #Oversampling only for the training data
    def __init__(self):
        self.scaler= StandardScaler()
        self.mean = None
        self.categorical_predictors = [
            'ps_ind_02_cat',
            'ps_ind_04_cat',
            'ps_ind_05_cat',
            'ps_ind_06_bin',
            'ps_ind_07_bin',
            'ps_ind_08_bin',
            'ps_ind_09_bin',
            'ps_ind_10_bin',
            'ps_ind_11_bin',
            'ps_ind_12_bin',
            'ps_ind_13_bin',
            'ps_ind_16_bin',
            'ps_ind_17_bin',
            'ps_ind_18_bin',
            'ps_car_01_cat',
            'ps_car_02_cat',
            'ps_car_03_cat', #400k+ NA however some kernel indicates this is not a trivial variable.
            'ps_car_04_cat',
            'ps_car_05_cat',
            'ps_car_06_cat',
            'ps_car_07_cat',
            'ps_car_08_cat',
            'ps_car_09_cat',
            'ps_car_10_cat',
            'ps_car_11_cat',   
            'ps_calc_15_bin',
            'ps_calc_16_bin',
            'ps_calc_17_bin',
            'ps_calc_18_bin',
            'ps_calc_19_bin',
            'ps_calc_20_bin'
            ]
        self.numeric_predictors = [
             'ps_ind_01',
             'ps_ind_03',
             'ps_ind_14',
             'ps_ind_15',
             'ps_reg_01',
             'ps_reg_02',
             'ps_reg_03',
             'ps_car_11',
             'ps_car_12',
             'ps_car_13',
             'ps_car_14',
             'ps_car_15',
             'ps_calc_01',
             'ps_calc_02',
             'ps_calc_03',
             'ps_calc_04',
             'ps_calc_05',
             'ps_calc_06',
             'ps_calc_07',
             'ps_calc_08',
             'ps_calc_09',
             'ps_calc_10',
             'ps_calc_11',
             'ps_calc_12',
             'ps_calc_13',
             'ps_calc_14',
             'na_sum',             #newly added
             'na_ca_t_sum'        #newly added
            ]
        
        
    def process(self, data,y =None, test= None, rso=1):
        data['na_sum'] = (data == -1).sum(axis=1)
        data['na_ca_t_sum'] = (data[self.categorical_predictors] == -1).sum(axis=1)
        
        #print(data['na_cat_sum'])
        data[data == -1] = np.nan
        for col in data.columns:
            if 'bin' in col or 'cat' in col:
                data[col] = data[col].astype('category')
        data = pd.get_dummies(data)
        if test == None:
            self.mean= data.loc[:,self.numeric_predictors].mean(axis= 0)
            data[self.numeric_predictors] =data[self.numeric_predictors].fillna(self.mean)
            self.scaler = self.scaler.fit(data)
            data = self.scaler.transform(data)
            sm = SMOTE(random_state=rso, ratio = 1)
            X_over, y_over = sm.fit_sample(data, y)
            return X_over, y_over
        if test == True:
            data[self.numeric_predictors] = data[self.numeric_predictors].fillna(self.mean)
            data = self.scaler.transform(data)

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