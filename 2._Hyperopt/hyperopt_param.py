# -*- coding:utf-8 -*-
import os
import pickle

import numpy as np
import pandas as pd
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ===============base_models_space==============
rbf_svc_space = {'C': hp.quniform('C', 0.01, 50, 0.01),
                 'gamma': hp.quniform('gamma', 0.001, 10, 0.001)
                 }

linear_svc_space = {'C': hp.quniform('C', 0.01, 50, 0.01),
                    }

RF_space = {'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500, 600, 700, 800, 900]),
            'criterion': hp.choice('criterion', ['gini', 'entropy']),
            'max_depth': hp.quniform('max_depth', 1, 15, 1),
            'min_samples_split': hp.quniform('min_samples_split', 0.01, 1, 0.01),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 0.01, 0.5, 0.01),
            'max_features':hp.quniform('max_features', 0.05, 1, 0.05),
            'class_weight': hp.choice('class_weight', ['balanced_subsample', 'balanced'])
            }


dt_space = {'criterion':hp.choice('dt_criterion',["gini", "entropy"]),
            'splitter':hp.choice('dt_splitter', ["best", "random"]),
            'max_depth':hp.quniform('dt_max_depth', 5, 20, 1),
            'min_samples_split':hp.quniform('dt_min_samples_split', 0.001, 0.2, 0.001),
            'min_samples_leaf': hp.quniform('dt_min_samples_leaf', 0.001, 0.2, 0.001),
            'max_features':hp.quniform('dt_max_features', 0.7, 1.0, 0.05),
            }

Adaboost_space = {#'base_estimator':DecisionTreeClassifier(),
                  'n_estimators': hp.choice('ada_n_estimators', [100, 200, 300, 400, 500, 600, 700]),
                  'learning_rate': hp.quniform('ada_learning_rate', 0.1, 1, 0.05),
                  'algorithm': hp.choice('ada_algorithm', ['SAMME', 'SAMME.R'])
                  }


ada_spaces = dict({'dt':dt_space, 'ada':Adaboost_space})


XBG_space = {'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500, 600, 700, 800]),
             'learning_rate': hp.quniform('learning_rate', 0.1, 1, 0.05),
             'max_depth': hp.choice("max_depth", np.arange(3, 19, 1)),
             'subsample': hp.quniform('subsample', 0.5, 1.0, 0.1),
             'min_child_weight': hp.quniform('min_child_weight', 1, 21, 1),
             'booster':hp.choice('booster', ['gbtree', 'gblinear']),
             'scale_pos_weight':hp.quniform('scale_pos_weight', 0.1, 1.0, 0.1),
             'gamma':hp.quniform('gamma', 0, 1, 0.05)
            }


# =======ensemble_models_space=======
eec_space = {'n_estimators': hp.choice('eec_n_estimators', np.arange(5, 16))}
EEC_Space = dict({'dt':dt_space, 'ada':Adaboost_space, 'eec':eec_space})

# =============loss_func=============
def loss_func_(f1, epsilon=1e-03):
    loss_score = - np.log(f1 + epsilon)
    # if label == 'label0.25':
    #     loss_score = - np.log(acc+epsilon)
    # elif label == 'label0.5':
    #     loss_score = - np.log(auc+epsilon)
    # elif label == 'label0.75':
    #     loss_score = - 0.7*np.log(recall+epsilon) - 0.3*np.log(acc+epsilon)
    # # harmonic_mean = 2*log_score1*log_score2 / (log_score1+log_score2)
    return loss_score


if __name__ == '__main__':
    exl_pth = os.path.abspath('data_processed.xlsx')
    train_set_p1 = pd.read_excel(exl_pth, sheet_name='train_set', dtype=float)
    # train_set_p2 = pd.read_excel(exl_pth, sheet_name='train_set_p2', dtype=float)

    label_binary = ['label0.25', 'label0.5', 'label0.75']
    three_label = ['three_clf']

    data = train_set_p1

    x = data.iloc[:, :33]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # # -------1.rbf_svc-------
    def object_function_rsvc(param):
        model_curr = SVC(**param, kernel='rbf', class_weight='balanced')                                   # **********
        score = cross_validate(model_curr, x_scaled, y, cv=10, scoring=['accuracy', 'roc_auc', 'balanced_accuracy', 'f1'], n_jobs=10)
        acc = score['test_accuracy'].mean()
        auc = score['test_roc_auc'].mean()
        acc_balanced = score['test_balanced_accuracy'].mean()
        f1 = score['test_f1'].mean()

        return {"loss": -acc_balanced, "status": STATUS_OK, 'scores': [acc, auc, f1]}

    for label in label_binary:
        print(f'{label}-rbf_svc'.center(40, '*'))                   # *********
        y = data[label]

        trails = Trials()
        best = fmin(
            fn=object_function_rsvc,
            space=rbf_svc_space,                                        # **********
            algo=tpe.suggest,
            max_evals=1000,
            trials=trails
        )
        print("Best: {}".format(best))
        print(min(trails.results, key=lambda keys:keys['loss']))

    # ----------3.RF-------------
    def object_function_rf(param):
        model_curr = RandomForestClassifier(**param)                        # ********
        score = cross_validate(model_curr, x_scaled, y, cv=10, scoring=['accuracy', 'roc_auc', 'balanced_accuracy', 'f1'], n_jobs=10)
        acc = score['test_accuracy'].mean()
        auc = score['test_roc_auc'].mean()
        acc_balanced = score['test_balanced_accuracy'].mean()
        f1 = score['test_f1'].mean()

        return {"loss": -acc_balanced, "status": STATUS_OK, 'scores': [acc, auc, f1]}

    for label in label_binary:
        print(f'{label}-RF'.center(40, '*'))                                   # *********
        y = data[label]

        trails = Trials()
        best = fmin(
            fn=object_function_rf,
            space=RF_space,                                                        # *********
            algo=tpe.suggest,
            max_evals=3000,
            trials=trails
        )
        print("Best: {}".format(best))
        print(min(trails.results, key=lambda keys:keys['loss']))

    # --------easy_ensemble--------
    def object_function_eec(param):
        dt_model = DecisionTreeClassifier(**param['dt'])
        ada_model = AdaBoostClassifier(base_estimator=dt_model, **param['ada'])
        model_curr = EasyEnsembleClassifier(base_estimator=ada_model, n_jobs=2)                                     # **********

        score = cross_validate(model_curr, x_scaled, y, cv=10, scoring=['accuracy', 'balanced_accuracy', 'roc_auc', 'f1'], n_jobs=10)
        acc = score['test_accuracy'].mean()
        auc = score['test_roc_auc'].mean()
        f1 = score['test_f1'].mean()
        balanced_acc = score['test_balanced_accuracy'].mean()

        return {"loss": -balanced_acc, "status": STATUS_OK, 'scores': [acc, auc, f1]}

    batch = 500
    max_iter = 1500
    num_count = 0

    for max_evals in range(batch, max_iter+1, batch):
        for label in ['label0.25', 'label0.5', 'label0.75']:
            print(f'{label}-EEC'.center(40, '*'))  # *********
            y = data[label]

            pth_pass = os.path.abspath(f'./EEC_model/eecmodel_{label}_{num_count-1}.pkl')
            pth_curr = os.path.abspath(f'./EEC_model/eecmodel_{label}_{num_count}.pkl')
            if os.path.exists(pth_pass):
                trails = pickle.load(open(pth_pass, "rb"))    # load model
            else:
                trails = Trials()

            best = fmin(
                fn=object_function_eec,
                space=EEC_Space,                                                            # *********
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trails
            )

            pickle.dump(trails, open(pth_curr, 'wb'))
            print("Best: {}".format(best))
            print(min(trails.results, key=lambda keys:keys['loss']))

        num_count += 1