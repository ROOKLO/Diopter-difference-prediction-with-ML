import os

import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

dr = DecisionTreeRegressor(max_depth=14, max_features=0.6, min_samples_leaf=0.001, min_samples_split=0.001, splitter='random')
adareg = AdaBoostRegressor(n_estimators=700, learning_rate=0.7,base_estimator=dr)

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

adareg.fit(x_train_scaled, y_train)
y_hat = adareg.predict(x_test_scaled)

r, p = stats.pearsonr(x=y_test, y=y_hat)
r2 = metrics.r2_score(y_true=y_test, y_pred=y_hat)
mse = metrics.mean_squared_error(y_true=y_test, y_pred=y_hat)
mae = metrics.mean_absolute_error(y_true=y_test, y_pred=y_hat)

print(f"r(p):{r}({p}) \n"
      f"r2  :{r2} \n"
      f"MSE :{mse} \n"
      f"MAE :{mae}")