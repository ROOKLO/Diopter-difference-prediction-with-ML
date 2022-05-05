import os

import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

model = SVR(C=0.73, gamma=0.007)

# -----data-------
data_pth = os.path.abspath('../data_processed.xlsx')
train_set = pd.read_excel(data_pth, sheet_name='train_set', dtype=float)
test_set = pd.read_excel(data_pth, sheet_name='test_set', dtype=float)

# ----label-----
label = ['post_SE_R']

x_train, y_train = train_set.iloc[:, :33], train_set[label].values.ravel()
x_test, y_test = test_set.iloc[:, :33], test_set[label].values.ravel()

# -----Standarscalar-----
scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

model.fit(x_train_scaled, y_train)
y_hat = model.predict(x_test_scaled)

r, p = stats.pearsonr(x=y_test, y=y_hat)
r2 = metrics.r2_score(y_true=y_test, y_pred=y_hat)
mse = metrics.mean_squared_error(y_true=y_test, y_pred=y_hat)
mae = metrics.mean_absolute_error(y_true=y_test, y_pred=y_hat)

print(f"r(p):{r}({p}) \n"
      f"r2  :{r2} \n"
      f"MSE :{mse} \n"
      f"MAE :{mae}")