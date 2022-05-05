import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from ranger import Ranger
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.cuda import amp
from torch.utils.data import DataLoader, TensorDataset

nn_space = {'nn':
                {'layer0':hp.choice('layer0', np.arange(64, 2049, 64)),
                'layer1':hp.choice('layer1', np.arange(64, 2049, 64)),
                'layer2':hp.choice('layer2', np.arange(64, 2049, 64)),
                'layer3':hp.choice('layer3', np.arange(64, 2049, 64)),
                'layer4':hp.choice('layer4', np.arange(64, 2049, 64)),
                # 'layer5':hp.choice('layer5', np.arange(64, 2049, 64)),
                'dropout':hp.uniform('dropout', 0, 0.6)},

            'batch_size': hp.choice('batch_size', np.arange(128, 1024, 64)),
            'learning_rate': hp.uniform('learning_rate', 4e-05, 0.01),
            'epoch': 300
            }


# ---------model---------
class MLP_net(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, **param):
        super(MLP_net, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dropout = torch.nn.Dropout(param['dropout'])

        self.hidden_0 = torch.nn.Linear(num_inputs, param['layer0'])
        self.bn0 = torch.nn.BatchNorm1d(param['layer0'])

        self.hidden_1 = torch.nn.Linear(param['layer0'], param['layer1'])  # hidden layer
        self.bn1 = torch.nn.BatchNorm1d(param['layer1'])

        self.hidden_2 = torch.nn.Linear(param['layer1'], param['layer2'])
        self.bn2 = torch.nn.BatchNorm1d(param['layer2'])

        self.hidden_3 = torch.nn.Linear(param['layer2'], param['layer3'])  # hidden layer
        self.bn3 = torch.nn.BatchNorm1d(param['layer3'])

        self.hidden_4 = torch.nn.Linear(param['layer3'], param['layer4'])  # hidden layer
        self.bn4 = torch.nn.BatchNorm1d(param['layer4'])

        # self.hidden_5 = torch.nn.Linear(param['layer4'], param['layer5'])  # hidden layer
        # self.bn5 = torch.nn.BatchNorm1d(param['layer5'])

        self.out = torch.nn.Linear(param['layer4'], out_features=num_outputs)  # output layer

    def forward(self, x):
        x = F.leaky_relu(self.hidden_0(x))
        x = self.dropout(self.bn0(x))

        x = F.leaky_relu(self.hidden_1(x))
        x = self.dropout(self.bn1(x))

        x = F.leaky_relu(self.hidden_2(x))
        x = self.dropout(self.bn2(x))

        x = F.leaky_relu(self.hidden_3(x))
        x = self.dropout(self.bn3(x))

        x = F.leaky_relu(self.hidden_4(x))
        x = self.dropout(self.bn4(x))

        # x = F.leaky_relu(self.hidden_5(x))
        # x = self.dropout(self.bn5(x))

        x = self.out(x)
        return x


def count_scores(true_label, pred_label, proba):
    acc = metrics.accuracy_score(true_label, pred_label)
    fpr, tpr, thresholds = metrics.roc_curve(true_label, proba)
    auc = metrics.auc(fpr, tpr)
    tn, fp, fn, tp = metrics.confusion_matrix(true_label, pred_label).ravel()
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    return acc, auc, sensitivity, specificity


def train_mlp(x_data, y_data, cv_num, **param):
    # total = cv_num * param['epoch']
    # with tqdm(total=total, unit='epoch') as pbar:

    mse_list = list()

    # total = param['epoch'] * cv_num
    # with tqdm(total=cv_num, position=1) as pbar1:
    kf = KFold(n_splits=cv_num, shuffle=True)
    for i, index in enumerate(kf.split(x_data)):

        net = MLP_net(num_inputs=35, num_outputs=1, **param['nn'])

        net.cuda(0)

        x_train, y_train = x_data.iloc[index[0], :], y_data.iloc[index[0]]
        x_test, y_test = x_data.iloc[index[1], :], y_data.iloc[index[1]]

        scaler = StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        x_train_ts = torch.as_tensor(x_train_scaled)
        y_train_ts = torch.as_tensor(y_train.values)

        x_test_ts = torch.as_tensor(x_test_scaled)

        # -----optimizer-----
        optimizer = Ranger(net.parameters(), lr= param['learning_rate'])
        # -----scaler----
        scaler = amp.GradScaler()

        # train
        for epoch in range(param['epoch']):
            net.train()
            epoch_loss = 0.0

            train_dataset = TensorDataset(x_train_ts, y_train_ts)
            train_loader = DataLoader(dataset=train_dataset, batch_size=int(param['batch_size']), shuffle=True, num_workers=2, pin_memory=True)
            for x, y in train_loader:
                x = x.to(device='cuda', dtype=torch.float32)
                y = y.to(device='cuda', dtype=torch.float32).view(-1, 1)

                with amp.autocast():                                ###
                    pred = net(x)
                    criterion = nn.MSELoss()
                    loss = criterion(pred, y)

                optimizer.zero_grad()
                scaler.scale(loss).backward()                       ###
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                scaler.step(optimizer)                              ###
                scaler.update()                                     ###

        with torch.no_grad():
            x_test_ts = x_test_ts.to(device='cuda', dtype=torch.float32)
            test_pred = net(x_test_ts).data.cpu().numpy()
            mse = metrics.mean_squared_error(y_true=y_test, y_pred=test_pred)

        mse_list.append(mse)
    mse_mean = np.mean(mse_list)

    return mse_mean


if __name__ == '__main__':
    exl_pth = os.path.abspath('../data_processed.xlsx')
    train_set_p1 = pd.read_excel(exl_pth, sheet_name='train_set', dtype=float)

    label_binary = ['label0.25', 'label0.5', 'label0.75']
    three_label = ['three_clf']
    reg_lable = ['post_SE_R']

    datas = train_set_p1

    x = datas.iloc[:, :33]
    for label in reg_lable:
        print(f'{label}-mlp'.center(40, '*'))  # *********
        y = datas[label]

        def object_function(param):
            mse = train_mlp(x_data=x, y_data=y, cv_num=5, **param)
            return {"loss": mse, "status": STATUS_OK}

        trails = Trials()
        best = fmin(
            fn=object_function,
            space=nn_space,
            algo=tpe.suggest,
            max_evals=2000,
            trials=trails
        )

        print("Best: {}".format(best))
        print(min(trails.results, key=lambda keys: keys['loss']))


