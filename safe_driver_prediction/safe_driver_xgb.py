# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:17:15 2017

@author: dell
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
import datetime

os.chdir('C:\\Users\\dell\\Desktop\\Safe driver')
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
cat_cols = [col for col in train.columns if '_cat' in col]

#one-hot
train = pd.get_dummies(train, columns=cat_cols)
test = pd.get_dummies(test, columns=cat_cols)

target = train['target']
trainID = train['id']
testID = test['id']
train.drop('id', axis=1, inplace=True)
train.drop('target', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

#----------------------------------------over-sampling
#ada = ADASYN()
#train_resampled, target_resampled = ada.fit_sample(train, target)   #long time 

ros = RandomOverSampler(random_state=66)
train_resampled, target_resampled = ros.fit_sample(train, target)


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, dtest=None):
    
    starttime = datetime.datetime.now()
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['target'].values)
#        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
             early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #fit
    alg.fit(dtrain[predictors], dtrain['target'],eval_metric='auc')
        
    #prediction on train set
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    endtime = datetime.datetime.now()   
    
    #output
    print("accuracy: ", metrics.accuracy_score(dtrain['target'].values, dtrain_predictions))
    print("AUC score:", metrics.roc_auc_score(dtrain['target'], dtrain_predprob))
    print("time spent: ", (endtime - starttime).seconds, "s")  

                
predictors = [x for x in train.columns if x not in ['target', 'id']]
xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=20,                    #deal with imbalaced data
        seed=666)
   
modelfit(xgb1, train, predictors, useTrainCV=False)

#accuracy:  0.87125091564
#AUC score: 0.865806590821
#time spent:  1119 s

