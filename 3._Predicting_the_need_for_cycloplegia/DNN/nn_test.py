import logging
import os
import time

import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from MLP_net import *


def nn_test(net, device, param_pth, data, label):
    x_data = data.iloc[:, :33]
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
        sig = torch.sigmoid(pred).data.cpu().numpy()

        roc_data[f'{label}_{key}_label'] = y
        roc_data[f'{label}_{key}_proba'] = sig.ravel()

        pre_label = (sig > 0.5).reshape(-1)
        acc = metrics.accuracy_score(y_true=y, y_pred=pre_label)
        auc = metrics.roc_auc_score(y_true=y, y_score=sig.reshape(-1))
        tn, fp, fn, tp = metrics.confusion_matrix(y_true=y, y_pred=pre_label).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

    logging.info(f'{label.title()} | ACC:{acc:.4f} AUC:{auc:.4f} sensi:{sensitivity:.4f} speci:{specificity:.4f} ')


if __name__ == '__main__':
    root = os.path.dirname(os.getcwd())
    excel_path = os.path.join(root, 'data_processed.xlsx')
    test_data = pd.read_excel(excel_path, sheet_name='test_set', dtype=float)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda')

    model = dict()
    model['label0.25'] = MLP_net_25(num_inputs=33, n_outputs=1)
    model['label0.5'] = MLP_net_25(num_inputs=33, n_outputs=1)
    model['label0.75'] = MLP_net_25(num_inputs=33, n_outputs=1)

    labels = ['label0.25', 'label0.5', 'label0.75']

    param_pth = dict()
    param_pth['label0.25'] = {'weighted': os.path.abspath('checkpoint_25.pth')}
    param_pth['label0.5'] = {'weighted': os.path.abspath('checkpoint_5.pth')}
    param_pth['label0.75'] = {'weighted': os.path.abspath('checkpoint_75.pth')}

    roc_data = dict()

    for label in labels:
        net = model[label]
        for key in param_pth[label]:
            pth_curr = param_pth[label][key]

            nn_test(net=net,
                    device=device,
                    param_pth= pth_curr,
                    data=test_data,
                    label=label)

    df = pd.DataFrame(roc_data)
    print(df)

