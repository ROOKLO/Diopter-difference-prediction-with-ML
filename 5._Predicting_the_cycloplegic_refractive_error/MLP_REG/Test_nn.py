import logging
import os
import time

import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from MLP_net import *


def nn_test(net, device, param_pth, train, test, label):
    x_train = train.iloc[:, :-6]
    x_test, y_test = test.iloc[:, :-6], test[label].values.ravel()

    scalar = StandardScaler().fit(x_train)
    x_test_scal = scalar.transform(x_test)

    net.to(device=device)
    net.load_state_dict(torch.load(param_pth))
    # logging.info(f'Network:\n'
    #              f'\t{net.num_inputs} input features\n'
    #              f'\t{net.num_outputs} output neurons(classes)\n')

    start = time.time()
    net.eval()

    with torch.no_grad():
        x_ts = torch.as_tensor(x_test_scal)
        x_ts = x_ts.to(device=device, dtype=torch.float32)
        test_pred = net(x_ts).data.cpu().numpy().ravel()
        test_mse = metrics.mean_squared_error(y_true=y_test, y_pred=test_pred)
        test_mae = metrics.mean_absolute_error(y_true=y_test, y_pred=test_pred)
        test_r2 = metrics.r2_score(y_true=y_test, y_pred=test_pred)
        test_r, p = stats.pearsonr(x=y_test, y=test_pred)

        true_and_pre = pd.DataFrame({'true':y_test, 'MLP_pre':test_pred})
        true_and_pre.to_csv('./mlp_true_pre.csv', index_label=None)

    #logging.info(f'MSE:{test_mse:.5f} MAE:{test_mae:.5f} r2:{test_r2:.5f} r:{test_r:.5f}')
    return test_mse, test_mae, test_r2, test_r, p

if __name__ == '__main__':
    excel_path = os.path.abspath('data_processed.xlsx')
    train_data = pd.read_excel(excel_path, sheet_name='train_set', dtype=float)
    test_data = pd.read_excel(excel_path, sheet_name='test_set', dtype=float)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda')
    net = MLP_net_reg(num_inputs=33, n_outputs=1, dropout=0.2)

    label = 'post_SE_R'

    results = list()

    param_pth = os.path.abspath(f'./checkpoints/REG_CP63/CP_epoch63.pth')

    mse, mae, r2, r, p = nn_test(net=net,
                              device=device,
                              param_pth=param_pth,
                              train=train_data,
                              test=test_data,
                              label=label)

    results.append([mse, mae, r2, r, p])
    results_df = pd.DataFrame(results, index=['63'], columns=['mse', 'mae', 'r2', 'r', 'p'])
    print(results_df.sort_values(by='mse', ascending=True))
