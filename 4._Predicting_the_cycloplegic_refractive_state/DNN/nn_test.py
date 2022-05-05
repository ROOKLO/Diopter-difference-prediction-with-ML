import logging
import os
import time

import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from MLP_net import *


def nn_test(net, device, param_pth, data, label):
    x_data = data.iloc[:, :-6]
    y = data[label]

    x_data_scaled = StandardScaler().fit_transform(x_data)

    net.to(device=device)
    net.load_state_dict(torch.load(param_pth))
    # logging.info(f'Network:\n'
    #              f'\t{net.num_inputs} input features\n'
    #              f'\t{net.num_outputs} output neurons(classes)\n')

    start = time.time()
    net.eval()
    with torch.no_grad():
        x_ts = torch.as_tensor(x_data_scaled)
        x_ts = x_ts.to(device=device, dtype=torch.float32)
        pred = net(x_ts)

        test_log_prob = torch.log_softmax(pred, dim=1)
        _, test_prelabel = torch.max(test_log_prob.data, 1)
        test_prelabel = test_prelabel.data.cpu().numpy()

        cm = metrics.confusion_matrix(y_true=y, y_pred=test_prelabel)

        clf_report = metrics.classification_report(y_true=y,
                                                   y_pred=test_prelabel,
                                                   # output_dict=True,
                                                   digits=6,
                                                   target_names=['0', '1', '2'])


        test_acc = metrics.accuracy_score(y_true=y, y_pred=test_prelabel)
        test_f1 = metrics.f1_score(y_true=y, y_pred=test_prelabel, average='macro')
        test_precision = metrics.precision_score(y_true=y, y_pred=test_prelabel, average='macro')
        test_recall = metrics.recall_score(y_true=y, y_pred=test_prelabel, average='macro')

    logging.info(f'{label.title()} | ACC:{test_acc:.4f} F1:{test_f1:.4f} precision:{test_precision:.4f} recall:{test_recall:.4f}')


if __name__ == '__main__':

    excel_path = os.path.abspath('../data_processed.xlsx')
    test_data = pd.read_excel(excel_path, sheet_name='test_set', dtype=float)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda')
    net = MLP_net_thri(num_inputs=33, n_outputs=3)

    label = 'three_clf'

    state_dict = dict()
    state_dict['three_way'] = os.path.abspath('checkpoint.pth')

    for key in state_dict:
        nn_test(net=net,
                device=device,
                param_pth= state_dict[key],
                data=test_data,
                label=label)