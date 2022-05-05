# -*- coding:utf-8 -*-
# Create_Time：2021/1/19 9:56
import numpy as np
import pandas as pd
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.utils import shuffle as Shuffle

excel_pth = os.path.abspath('data_raw.xlsx')

# ====================imputation & split============================
# -----data------
data = pd.read_excel(excel_pth, dtype=float)
x_data, rest_data = data.iloc[:, :33], data.iloc[:, 33:]
r, c = x_data.shape[0], x_data.shape[1]
column_names = x_data.columns

# -----imputation-----
bayes = BayesianRidge(n_iter=300, )
imputer = IterativeImputer(estimator=bayes, max_iter=100, missing_values=np.nan)
data_imped = imputer.fit_transform(x_data)
data_df = pd.DataFrame(data_imped, columns=column_names)

# -----split-----
data_final = pd.concat([data_df, rest_data], axis=1)
data_sf = Shuffle(data_final)
test_size = int(len(data_sf) * (ratio := 0.1))
test_set_ = data_sf.iloc[:test_size, :]
train_set_ = data_sf.iloc[test_size:, :]

# -----save-----
with pd.ExcelWriter('data_processed.xlsx') as writer:
    train_set_.to_excel(writer, sheet_name='train_set', index=False)
    test_set_.to_excel(writer, sheet_name='test_set', index=False)