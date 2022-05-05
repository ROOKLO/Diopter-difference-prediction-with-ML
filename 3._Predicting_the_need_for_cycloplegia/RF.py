# -*- coding:utf-8 -*-

import os
import time

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def count_scores(true_label, pred_label, proba):
    acc = metrics.accuracy_score(true_label, pred_label)
    fpr, tpr, thresholds = metrics.roc_curve(true_label, proba)
    auc = metrics.auc(fpr, tpr)
    tn, fp, fn, tp = metrics.confusion_matrix(true_label, pred_label).ravel()
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    return acc, auc, sensitivity, specificity

rf_25 = RandomForestClassifier(n_estimators=400,
                               criterion='entropy',
                               max_depth=12,
                               max_features=0.5,
                               min_samples_leaf=0.01,
                               min_samples_split=0.01,
                               n_jobs=-2,
                               class_weight='balanced_subsample'
                               )

rf_5 = RandomForestClassifier(n_estimators=300,
                              criterion='entropy',
                              max_depth=8,
                              max_features=1.0,
                              min_samples_leaf=0.01,
                              min_samples_split=0.01,
                              n_jobs=-2,
                              class_weight='balanced',
                              )

rf_75 = RandomForestClassifier(n_estimators=300,
                               criterion='entropy',
                               max_depth=15,
                               max_features=0.95,
                               min_samples_leaf=0.01,
                               min_samples_split=0.02,
                               n_jobs=-2,
                               class_weight='balanced_subsample'
                               )

rf_model = dict({'label0.25':rf_25, 'label0.5':rf_5, 'label0.75':rf_75})

# ------label-----
labels = ['label0.25', 'label0.5', 'label0.75']

# -----data-------
data_pth = os.path.abspath('../data_processed.xlsx')
train_set = pd.read_excel(data_pth, sheet_name='train_set', dtype=float)
test_set = pd.read_excel(data_pth, sheet_name='test_set', dtype=float)


if __name__ == '__main__':
    acc_means = list()
    auc_means = list()
    sensi_means = list()
    speci_means = list()

    index = list()
    result = dict()

    start = time.time()

    iter_num = 1

    total = len(labels) * iter_num
    with tqdm(total=total) as pbar:

        for label in labels:
            model = rf_model[label]

            acc_list = list()
            auc_list = list()
            sensi_list = list()
            speci_list = list()

            proba_list, y_test_list = list(), list()

            for _ in range(iter_num):
                x_train, y_train = train_set.iloc[:, :33], train_set[label]
                x_test, y_test = test_set.iloc[:, :33], test_set[label]

                # -----Standarscalar-----
                scaler = StandardScaler().fit(x_train)
                x_train_scaled = scaler.transform(x_train)
                x_test_scaled = scaler.transform(x_test)

                model.fit(x_train_scaled, y_train)
                prelabel = model.predict(x_test_scaled)

                if model.__class__.__name__ == 'SVC':
                        score = model.decision_function(x_test_scaled)
                if model.__class__.__name__ != 'SVC':
                        score_ = model.predict_proba(x_test_scaled)
                        score = score_[:, 1]

                y_test_list += list(y_test)
                proba_list += list(score)

                acc, auc, sensitivity, specificity = count_scores(y_test, prelabel, score)

                acc_list.append(acc)
                auc_list.append(auc)
                sensi_list.append(sensitivity)
                speci_list.append(specificity)

                pbar.update(1)

            index.append(f'RF-{label}')

            acc_means.append(np.mean(acc_list))
            auc_means.append(np.mean(auc_list))
            sensi_means.append(np.mean(sensi_list))
            speci_means.append(np.mean(speci_list))

        result['ACC'] = acc_means
        result['AUC'] = auc_means
        result['sensitivity'] = sensi_means
        result['specificity'] = speci_means

        result_df = pd.DataFrame(result, index=index)
        print(result_df)
        print(f'TIME ELAPSE:{(time.time() - start) / 3600}h')
