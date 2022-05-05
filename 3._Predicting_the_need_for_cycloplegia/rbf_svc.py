# -*- coding:utf-8 -*-

import os
import time

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


def count_scores(true_label, pred_label, proba):
    acc = metrics.accuracy_score(true_label, pred_label)
    fpr, tpr, thresholds = metrics.roc_curve(true_label, proba)
    auc = metrics.auc(fpr, tpr)
    tn, fp, fn, tp = metrics.confusion_matrix(true_label, pred_label).ravel()
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    return acc, auc, sensitivity, specificity


rbf_25 = SVC(C=22.15, kernel='rbf', gamma=0.014, class_weight='balanced', probability=False)
rbf_5 = SVC(C=9.58, kernel='rbf', gamma=0.004, class_weight='balanced', probability=False)
rbf_75 = SVC(C=21.03, kernel='rbf', gamma=0.001, class_weight='balanced', probability=False)


rbf_svc_model = dict({'label0.25':rbf_25, 'label0.5':rbf_5, 'label0.75':rbf_75})

# ------label-----
labels = ['label0.25', 'label0.5', 'label0.75']

# -----data-------
data_pth = os.path.abspath('data_processed.xlsx')
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
            model = rbf_svc_model[label]

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

                proba_list  += list(score)
                y_test_list += list(y_test)

                acc, auc, sensitivity, specificity = count_scores(y_test, prelabel, score)

                acc_list.append(acc)
                auc_list.append(auc)
                sensi_list.append(sensitivity)
                speci_list.append(specificity)

                pbar.update(1)

            index.append(f'rbf_svc-{label}')

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