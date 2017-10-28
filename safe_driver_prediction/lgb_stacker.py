import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=False)

lgb1_params = {
    'boosting_type':'gbdt',
    'objective':'binary',
    'metric': {'l2','binary_logloss'},
    'learning_rate': 0.05, 
    'feature_fraction': 0.9,   #
    'bagging_fraction': 0.8,   #
    'bagging_freq': 5,
    'verbose': 0
}

lgb1_hyper = {'params': lgb1_params,
             'train_set': None
              'valid_sets': None
              'num_boost_round': 1500, 
              'early_stopping_rounds': 5
             }

# lgb1 = lgb.train(lgb1_params,
#         lgb_train,
#         num_boost_round =1500,  #Number of trees it will generate  
#         valid_sets= lgb_val,
#     early_stopping_rounds = 5
#         )

lgb2_params = {
    'boosting_type':'gbdt',
    'objective':'binary',
    'metric': {'l2','binary_logloss'}, 
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'feature_freq' :10,
    'num_leaves':24 
}

lgb2_hyper = {'params': lgb2_params,
             'train_set': None
              'valid_sets': None
              'num_boost_round': 2500, 
              'learning_rates': lambda x: 0.05*(0.999**x), 
              'early_stopping_rounds': 5
             }

# lgb2 = lgb.train(lgb2_params,
#         lgb_train,
#         valid_sets= lgb_val,
#     num_boost_round = 2500,
#     learning_rates= lambda x: 0.05*(0.999**x), #step_decay
#     early_stopping_rounds = 5
#         )

lgb3_params = {
    'boosting_type':'gbdt', 
    'objective':'binary',
    'metric': {'l2','binary_logloss'},
    'learning_rate': 0.03, 
    'feature_fraction': 0.8, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 10, 
    'max_depth':6 
}

lgb3_hyper = {'params': lgb3_params,
             'train_set': None
              'valid_sets': None
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
        
    def fit_predict(self, X, y, T, cv = True):
        Final_score_test = np.zeros((T.shape[0], self.num_base_models))
        val_score_test = np.zeros((X.shape[0], self.num_base_models))
        i = 0

        for lgb_hyper in self.base_models.get('lgb'):
            folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))
            for j, (train_idx, val_idx) in enumerate(folds) :
                lgb_hyper['train_set'] = lgb.Dataset(X.iloc[train_idx, :], 
                                            y.iloc[train_idx], 
                                            free_raw_data=False)
                lgb_hyper['valid_sets'] = lgb.Dataset(X.iloc[val_idx, :], 
                                                      y_val = y.iloc[val_idx], 
                                                      reference=lgb_hyper['train_set'], 
                                                      free_raw_data=False)
                
                
                lgb = lgb.train(**lgb_hyper)
                val_score_test[val_idx, i] += lgb.predict(lgb_hyper['valid_sets'], lgb.best_iteration_)
                Final_score_test[:, i] = lgb.predict(T, lgb.best_iteration_)
            val_score_test[:, i] /= self.n_splits
            Final_score_test[:, i] /= self.n_splits
            i += 1
        
        results = cross_val_score(self.stacker, val_score_test[:, i], y, cv=3, scoring='roc_auc')
        print(results)
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(val_score_test, y)
        res = self.stacker.predict_proba(Final_score_test)[:,1]
        return res


        
        
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