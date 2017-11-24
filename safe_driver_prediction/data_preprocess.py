from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
import sys

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
            'ps_ind_02_cat', #700
            'ps_ind_04_cat', #450
            'ps_ind_05_cat', #1000
            'ps_ind_06_bin', #380
            'ps_ind_07_bin', #550
            'ps_ind_08_bin', #200
            'ps_ind_09_bin', #200
            'ps_ind_10_bin', #1
            'ps_ind_11_bin', #1
            'ps_ind_12_bin', #16
            'ps_ind_13_bin', #4
            'ps_ind_16_bin', #430
            'ps_ind_17_bin', #580
            'ps_ind_18_bin', #44
            'ps_car_01_cat', #1000
            'ps_car_02_cat', #210
            'ps_car_03_cat', #300    #400k+ NA however some kernel indicates this is not a trivial variable.
            'ps_car_04_cat', #150
            'ps_car_05_cat', #385
            'ps_car_06_cat', #300
            'ps_car_07_cat', #372
            'ps_car_08_cat', #132
            'ps_car_09_cat', #700
            'ps_car_10_cat', #22
            'ps_car_11_cat', #450
            'ps_calc_15_bin', #78
            'ps_calc_16_bin', #236
            'ps_calc_17_bin', #400
            'ps_calc_18_bin', #320
            'ps_calc_19_bin', #400
            'ps_calc_20_bin' #70
            ]
        self.numeric_predictors = [
             'ps_ind_01', #2010
             'ps_ind_03', #2042
             'ps_ind_14', #17
             'ps_ind_15', #2374
             'ps_reg_01', #1479
             'ps_reg_02', #2335
             'ps_reg_03', #702
             'ps_car_11', #202
             'ps_car_12', #178
             'ps_car_13', #1140
             'ps_car_14', #406
             'ps_car_15', #721
             'ps_calc_01', #626
             'ps_calc_02', #731
             'ps_calc_03', #569
             'ps_calc_04', #599
             'ps_calc_05', #787
             'ps_calc_06', #851
             'ps_calc_07', #447
             'ps_calc_08', #314
             'ps_calc_09', #901
             'ps_calc_10', #289
             'ps_calc_11', #397
             'ps_calc_12', #802
             'ps_calc_13', #463
             'ps_calc_14', #338
             'na_sum',     #468        #newly added
             'na_ca_t_sum', #135       #newly added
             #'na_ca_t_bi_n_sum' not effective
            'ps_car_13_x_ps_reg_03'
            ]
        self.uselessones = ['ps_ind_14',
             'ps_ind_10_bin',
             'ps_ind_11_bin',
             'ps_ind_12_bin', 
             'ps_ind_13_bin',
             'ps_ind_18_bin',
             'ps_car_10_cat',
             'ps_calc_15_bin',
             'ps_calc_20_bin'
        ]
        
    def process(self, data, y=None, test= None, rso=1, variable_selection= False, oversample = None, norm_all =None):
        data['na_sum'] = (data == -1).sum(axis=1)
        data['na_ca_t_sum'] = (data[self.categorical_predictors] == -1).sum(axis=1)
        #data['na_ca_t_bi_n_sum'] = (data[[i for i in self.categorical_predictors if 'bin' in i] ] == -1).sum(axis=1)
        
        if variable_selection:
            data = data.drop(self.uselessones, axis=1)
        data[data == -1] = np.nan
        data['ps_car_13_x_ps_reg_03'] = data['ps_car_13'] * data['ps_reg_03']
        for col in data.columns:
            if 'bin' in col or 'cat' in col:#and col not in ['ps_reg_01_plus_ps_car_02_cat','ps_reg_01_plus_ps_car_04_cat']:
                data[col] = data[col].astype('category')
        data = pd.get_dummies(data)
        col = data.columns
        self.numeric_predictors = [i for i in self.numeric_predictors if i not in self.uselessones or not variable_selection]
        if test == None:
            self.mean= data.loc[:,self.numeric_predictors].mean(axis=0)
            #self.mean= data.loc[:,self.numeric_predictors + self.categorical_predictors].mean(axis= 0)
            data[self.numeric_predictors] =data[self.numeric_predictors].fillna(self.mean)
            if norm_all:
                self.scaler = self.scaler.fit(data)
                data = self.scaler.transform(data)
            else:
                self.scaler = self.scaler.fit(data[self.numeric_predictors])
                data[self.numeric_predictors] = self.scaler.transform(data[self.numeric_predictors])
                
            if oversample is not None:
                #sm = SMOTE(random_state=rso, ratio = oversample)
                #X_over, y_over = sm.fit_sample(data, y)
                ros = RandomOverSampler(random_state=0, ratio=oversample)
                X_over, y_over = ros.fit_sample(data, y)
                return X_over, y_over, col
            return data, y, col
        if test == True:
            data[self.numeric_predictors] =data[self.numeric_predictors].fillna(self.mean)
            if norm_all:
                self.scaler = self.scaler.fit(data)
                data = self.scaler.transform(data)
            else:
                self.scaler = self.scaler.fit(data[self.numeric_predictors])
                data[self.numeric_predictors] = self.scaler.transform(data[self.numeric_predictors])
        return data    
        
        
        


