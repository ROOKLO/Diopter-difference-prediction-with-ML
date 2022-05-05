# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# ----------param_space-----------
ext_space = {'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500, 600, 700, 800, 900]),
             'max_depth': hp.quniform('max_depth', 1, 50, 1),
             'min_samples_split': hp.choice('min_samples_split', np.arange(2, 400, 2)),
             'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 50, 1)),
             'max_features':hp.quniform('max_features', 0.05, 1, 0.05),
             # 'max_samples':hp.quniform('max_samples', 0.6, 0.99, 0.01)
            }


XGB_space = {'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500, 600, 700, 800]),      
             'learning_rate': hp.quniform('learning_rate', 0.1, 1, 0.05),                            
             'max_depth': hp.choice("max_depth", np.arange(3, 30, 1)),                               
             'subsample': hp.quniform('subsample', 0.1, 1.0, 0.1),                                   
             'min_child_weight': hp.quniform('min_child_weight', 1, 21, 1),                          
             'booster':hp.choice('booster', ['gbtree', 'gblinear']),                                
             'scale_pos_weight':hp.quniform('scale_pos_weight', 0.1, 1.0, 0.1),                       
             'gamma':hp.quniform('gamma', 0, 1, 0.05)                                               
            }

# -----------ada_reg_space---------------
dt_space = {'splitter':hp.choice('dt_splitter', ["best", "random"]),                                 
            'max_depth':hp.quniform('dt_max_depth', 7, 14, 1),                                        
            'min_samples_split':hp.quniform('dt_min_samples_split', 0.001, 0.2, 0.001),               
            'min_samples_leaf': hp.quniform('dt_min_samples_leaf', 0.001, 0.2, 0.001),                
            'max_features':hp.quniform('dt_max_features', 0.5, 1.0, 0.05),                          
            }                                                                                          


Adaboost_space = {
                  'n_estimators': hp.choice('ada_n_estimators', [100, 200, 300, 400, 500, 600, 700]),  
                  'learning_rate': hp.quniform('ada_learning_rate', 0.1, 1, 0.05),                    
                  }
ada_spaces = dict({'dt':dt_space, 'ada':Adaboost_space})


# ----------SVR--------------
rbf_svr_space = {'C': hp.quniform('C', 0.01, 50, 0.01),             # 5000
                 'gamma': hp.quniform('gamma', 0.001, 10, 0.001)     # 10000
                 }

# ----------RF---------------
RFr_space = {'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500, 600, 700, 800, 900]), 
            'max_depth': hp.quniform('max_depth', 1, 15, 1),                                          
            'min_samples_split': hp.quniform('min_samples_split', 0.01, 1, 0.01),                     
            'min_samples_leaf': hp.quniform('min_samples_leaf', 0.01, 0.5, 0.01),                     
            'max_features':hp.quniform('max_features', 0.05, 1, 0.05),                               
            }

# -------------data----------------
data_pth = os.path.abspath('../data_processed.xlsx')
data = pd.read_excel(data_pth, sheet_name='train_set', dtype=float)

label = ['post_SE_R']

x, y = data.iloc[:, :-6], data[label].values.ravel()
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# ----------RF-----------
def object_function_RF(param):
    model_curr = RandomForestRegressor(**param)                                   # **********
    score = cross_validate(model_curr, x_scaled, y, cv=10, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                           n_jobs=10)
    r2_mean = score['test_r2'].mean()
    mse_mean = score['test_neg_mean_squared_error'].mean()
    mae_mean = score['test_neg_mean_absolute_error'].mean()

    return {"loss": -r2_mean, "status": STATUS_OK, 'scores': [mse_mean, mae_mean]}

trails = Trials()
best = fmin(
        fn=object_function_RF,
        space=RFr_space,                                        # **********
        algo=tpe.suggest,
        max_evals=3000,
        trials=trails
)
print("Best: {}".format(best))
print(min(trails.results, key=lambda keys:keys['loss']))


# --------rbf_svc--------
def object_function_svr(param):
    model_curr = SVR(**param)                                   # **********
    score = cross_validate(model_curr, x_scaled, y, cv=10, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                           n_jobs=10)
    r2_mean = score['test_r2'].mean()
    mse_mean = score['test_neg_mean_squared_error'].mean()
    mae_mean = score['test_neg_mean_absolute_error'].mean()

    return {"loss": -r2_mean, "status": STATUS_OK, 'scores': [mse_mean, mae_mean]}

trails = Trials()
best = fmin(
        fn=object_function_svr,
        space=rbf_svr_space,                                        # **********
        algo=tpe.suggest,
        max_evals=1000,
        trials=trails
)
print("Best: {}".format(best))
print(min(trails.results, key=lambda keys:keys['loss']))

# ------------Ada_reg--------------
def object_function_adareg(param):
    dtreg = DecisionTreeRegressor(**param['dt'])
    model_curr = AdaBoostRegressor(base_estimator=dtreg ,**param['ada'])                                   # **********
    score = cross_validate(model_curr, x_scaled, y, cv=10, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                           n_jobs=10)
    r2_mean = score['test_r2'].mean()
    mse_mean = score['test_neg_mean_squared_error'].mean()
    mae_mean = score['test_neg_mean_absolute_error'].mean()

    return {"loss": -r2_mean, "status": STATUS_OK, 'scores': [mse_mean, mae_mean]}

trails = Trials()
best = fmin(
        fn=object_function_adareg,
        space=ada_spaces,                                        # **********
        algo=tpe.suggest,
        max_evals=3000,
        trials=trails
)
print("Best: {}".format(best))
print(min(trails.results, key=lambda keys:keys['loss']))
