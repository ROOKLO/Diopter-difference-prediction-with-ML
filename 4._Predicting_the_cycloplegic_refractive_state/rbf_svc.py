
import os
import time

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

rbf = SVC(C=2.3, gamma=0.008, class_weight='balanced' ,probability=True)

# -----data-------
data_pth = os.path.abspath('data_processed.xlsx')
train_set = pd.read_excel(data_pth, sheet_name='train_set', dtype=float)
test_set = pd.read_excel(data_pth, sheet_name='test_set', dtype=float)

# ----label-----
label = ['three_clf']

if __name__ == '__main__':
    acc_means = list()
    precision_means = list()
    recall_means = list()
    f1_means = list()

    index = list()
    result = dict()

    start = time.time()

    iter_num = 1

    total = iter_num
    with tqdm(total=total) as pbar:

        model = rbf

        acc_list = list()
        precision_list = list()
        recall_list = list()
        f1_list = list()

        for _ in range(iter_num):
            x_train, y_train = train_set.iloc[:, :33], train_set[label].values.ravel()
            x_test, y_test = test_set.iloc[:, :33], test_set[label]

            # -----Standarscalar-----
            scaler = StandardScaler().fit(x_train)
            x_train_scaled = scaler.transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            model.fit(x_train_scaled, y_train)
            prelabel = model.predict(x_test_scaled)

            score = model.predict_proba(x_test_scaled)
            # score = score_[:, 1]

            clf_report = metrics.classification_report(y_true=y_test,
                                                       y_pred=prelabel,
                                                       output_dict=True,
                                                       digits = 6,
                                                       target_names=['0', '1','2'])

            acc = clf_report['accuracy']
            precision_ = clf_report['macro avg']['precision']
            recall_ = clf_report['macro avg']['recall']
            f1_ = clf_report['macro avg']['f1-score']

            acc_list.append(acc)
            recall_list.append(recall_)
            precision_list.append(precision_)
            f1_list.append(f1_)

            pbar.update(1)

        index.append(f'rbf_svc-three_way') # index

        acc_means.append(np.mean(acc_list))
        precision_means.append(np.mean(precision_list))
        recall_means.append(np.mean(recall_list))
        f1_means.append(np.mean(f1_list))

    result['ACC'] = acc_means
    result['precision'] = precision_means
    result['recall'] = recall_means
    result['f1_score'] = f1_means

    result_df = pd.DataFrame(result, index=index)
    print(result_df)