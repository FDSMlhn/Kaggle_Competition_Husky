import pandas as pd
import numpy as np
import data_util

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

lgb1_params = {
    'boosting_type':'gbdt',
    'objective':'binary',
    'metric': {'binary_logloss'},
    'learning_rate': 0.05, 
    'feature_fraction': 0.9,   #
    'bagging_fraction': 0.8,   #
    'bagging_freq': 5,
    'verbose': 0,
    'lambda_l2':0.0342
}

lgb1_hyper = {'params': lgb1_params,
             'train_set': None,
              'valid_sets': None,
              'num_boost_round': 1500, 
              'early_stopping_rounds': 5,
                  'verbose_eval':50
             }


lgb2_params = {
    'boosting_type':'gbdt',
    'objective':'binary',
    'metric': {'binary_logloss'}, 
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'num_leaves':24,
    'lambda_l2':0.0342
}

lgb2_hyper = {'params': lgb2_params,
             'train_set': None,
              'valid_sets': None,
              'num_boost_round': 2500, 
              'learning_rates': lambda x: 0.05*(0.999**x), 
              'early_stopping_rounds': 5
             }


lgb3_params = {
    'boosting_type':'gbdt', 
    'objective':'binary',
    'metric': {'binary_logloss'},
    'learning_rate': 0.03, 
    'feature_fraction': 0.8, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 10, 
    'max_depth':6,
    'lambda_l2':0.0342
}

lgb3_hyper = {'params': lgb3_params,
             'train_set': None,
              'valid_sets': None,
              'num_boost_round': 2000, 
              'early_stopping_rounds': 10
             }

# lgb3 = lgb.train(lgb3_params,
#         lgb_train,
#         valid_sets= lgb_val,
#     num_boost_round = 2000,
#     early_stopping_rounds = 10
#         )

# lgb1 = lgb.cv(lgb1_params,
#         lgb_train,
#         num_boost_round =1250,  #Number of trees it will generate  
#         valid_sets= lgb_val,
#     early_stopping_rounds = 5
#         )

stack_model = LogisticRegression()
all_base_models = {'lgb': [lgb1_hyper,lgb2_hyper,lgb3_hyper]}


class lgb_ensemble():
    def __init__(self, n_splits, num_base_models, stacker= stack_model, base_models= all_base_models):
        self.n_splits = n_splits
        self.num_base_models = num_base_models
        self.stacker= stacker
        self.base_models= base_models
        
    def fit_predict(self, X, y, T, META = None, rso=66):
        miscellany= {}
        
        S_train = np.zeros((X.shape[0], self.num_base_models))
        S_test  = np.zeros((T.shape[0], self.num_base_models))
        i = 0
        
        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=rso).split(X, y))
        
        for lgb_hyper in self.base_models.get('lgb'):
            S_test_i =  np.zeros((T.shape[0], self.n_splits))
            
            for j, (train_idx, val_idx) in enumerate(folds) :
                lgb_hyper['train_set'] = lgb.Dataset(X[train_idx, :], 
                                                     y[train_idx], 
                                                     free_raw_data=False)
                lgb_hyper['valid_sets'] = lgb.Dataset(X[val_idx, :], 
                                                      y[val_idx], 
                                                      reference=lgb_hyper['train_set'], 
                                                      free_raw_data=False)

                lgb_m = lgb.train(**lgb_hyper)
                S_train[val_idx, i] = lgb_m.predict(X[val_idx, :], lgb_m.best_iteration)
                S_test_i[:, j] = lgb_m.predict(T, lgb_m.best_iteration)
                
            #mean_idx = (temp_val_score!=0).sum(1)
            #mean_idx[mean_idx==0] = 1
            #val_score_test[:, i] = np.true_divide(temp_val_score.sum(axis=1), mean_idx)
            S_test[:, i] = S_test_i.mean(axis=1)
            i += 1
        
        miscellany['lgb_cor']= cor_mat(S_train)
        train_auc = []
        train_gini = []
        for m in range(i):
            train_auc.append(metrics.roc_auc_score(y,S_train[:,m]))
            train_gini.append(data_util.eval_gini(y,S_train[:,m]))
        miscellany['train_auc']  = train_auc
        miscellany['train_gini'] = train_gini
        
        
        if META == 'lgb':

            X_copy = np.hstack((X, S_train))
            T_copy = np.hstack((T, S_test))
            X_train, X_val, y_train, y_val = train_test_split(X_copy, y ,test_size =0.2 ,random_state=rso)
            lgb_hyper = self.base_models['lgb'][0]
            lgb_hyper['train_set'] = lgb.Dataset(X_train, 
                                                     y_train, 
                                                     free_raw_data=False)
            lgb_hyper['valid_sets'] = lgb.Dataset(X_val, 
                                                      y_val, 
                                                      reference=lgb_hyper['train_set'], 
                                                      free_raw_data=False)
            lgb_m1 = lgb.train(**lgb_hyper)
            
            y_pred = lgb_m1.predict(X_val, lgb_m1.best_iteration)
            res = lgb_m1.predict(T, lgb_m.best_iteration)
            
            result = metrics.roc_auc_score(y_val,y_pred)
            gini = data_util.eval_gini(y_val,y_pred)

            print("Stacker score: {}. Gini score: {}.".format(result, gini))

            return res, miscellany
        
        if META == 'lg':
            X_copy = np.hstack((X, val_score_test))
            T_copy = np.hstack((T, Final_score_test))
            
            

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print(results)
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res, miscellany


def cor_mat(x):
    mean_x = x- np.mean(x, axis=1, keepdims= True)
    sq_x = np.sum(mean_x**2, axis=0, keepdims = True)
    cor = np.dot(mean_x.T, mean_x) / np.sqrt(np.dot(sq_x.T, sq_x))
    return cor
    
        
# lgb2_params = {
#     'boosting_type':'gbdt',
#     'objective':'binary',
#     'metric': 'binary_logloss', 
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.6,
#     'bagging_freq': 5,
#     'feature_freq' :10,
#     'num_leaves':16  
# }

# lgb2 = lgb.train(lgb2_params,
#         lgb_train,
#         valid_sets= lgb_val,
#     num_boost_round = 2500,
#     learning_rates= lambda x: 0.05*(0.999**x), #step_decay
#     early_stopping_rounds = 5
#         )

# lgb3_params = {
#     'boosting_type':'gbdt',
#     'objective':'binary',
#     'metric': 'binary_logloss',
#     'learning_rate': 0.02,
#     'feature_fraction': 0.5,
#     'bagging_fraction': 0.6,
#     'bagging_freq': 5,
#     'max_depth':4
# }

# lgb3 = lgb.train(lgb3_params,
#         lgb_train,
#         valid_sets= lgb_val,
#     num_boost_round = 800,
#     early_stopping_rounds = 10
#         )

# y_pred1 = lgb1.predict(X_train_test)
# y_pred2 = lgb2.predict(X_train_test)
# y_pred3 = lgb3.predict(X_train_test)



# lgb_params_1 = {
#     'learning_rate': 0.01,
#     'n_estimators': 1250,
#     'max_bin': 10,
#     'subsample': 0.8,
#     'subsample_freq': 10,
#     'colsample_bytree': 0.8,   #feature_fraction
#     'min_child_samples': 500
# }

# lgb_params_2 = {
#     'learning_rate': 0.005,
#     'n_estimators': 3700,
#     'subsample': 0.7,
#     'subsample_freq': 2,
#     'colsample_bytree': 0.3,  
#     'num_leaves': 16
# }

# lgb_params_3 = {
#     'learning_rate': 0.02,
#     'n_estimators': 800,
#     'max_depth': 4
# }