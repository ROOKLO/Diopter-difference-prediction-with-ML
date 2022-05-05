import os
import logging
import os
import time

import numpy as np
import pandas as pd
import torch.nn as nn
from ranger import Ranger
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from torch.cuda import amp
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from Focal_Loss import *
from MLP_net import *

dir_checkpoint = 'checkpoints/'


def train_fc_net(net, device, train, test, label, epochs, batch_size, lr_init, save_cp):

    x_train, y_train = train.iloc[:, :-6], OneHotEncoder().fit_transform(train[label].values.reshape(-1, 1)).toarray()
    x_test, y_test = test.iloc[:, :-6], OneHotEncoder().fit_transform(test[label].values.reshape(-1, 1)).toarray()
    
    # ----test_set-------
    scaler_2 = StandardScaler().fit(x_train)
    x_test_scaled = scaler_2.transform(x_test)
    x_test_ts = torch.as_tensor(x_test_scaled)

    # ----train_set------
    x_train_, x_val, y_train_, y_val = train_test_split(x_train,
                                                        y_train,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        stratify=y_train,
                                                        random_state=7)

    scaler_1 = StandardScaler().fit(x_train_)
    x_train_scaled = scaler_1.transform(x_train_)
    x_val_scaled = scaler_1.transform(x_val)

    x_train_ts = torch.as_tensor(x_train_scaled)
    y_train_ts = torch.as_tensor(y_train_)

    y_val = torch.as_tensor(y_val)
    y_val_ts = y_val.to(device)
    y_test = torch.as_tensor(y_test)
    y_test_ts = y_test.to(device)

    train_dataset = TensorDataset(x_train_ts, y_train_ts)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # ------val_set------
    x_val_ts = torch.as_tensor(x_val_scaled)

    writer = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}_{label}/acc')
    val_writer_auc = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}_{label}/val_f1')
    val_writer_sensi = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}_{label}/val_precision')
    val_writer_speci = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}_{label}/val_recall')

    writer_auc = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}_{label}/test_f1')
    writer_sensi = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}_{label}/test_precision')
    writer_speci = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}_{label}/test_recall')

    global_step = 0                             # 1batch = 1step
    logging.info(f'''Starting training:
                    Epochs:         {epochs}
                    Batch_size:     {batch_size}
                    learning rate:  {lr_init}
                    Training size:  {int(len(train)*0.8)}
                    Validation size:{int(len(train)*0.2)}
                    Checkpoints:    {save_cp}
                    Device:         {device.type}
    ''')

    optimizer = Ranger(net.parameters(), lr=lr_init, weight_decay=1e-08)

    schedule = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                  base_lr=lr_init,
                                                  max_lr=lr_init*5,
                                                  step_size_up=625,       # 20*(train_size/batch_size)
                                                  step_size_down=625,
                                                  mode='triangular2',
                                                  cycle_momentum=False)

    # ----amp scaler-----
    scaler = amp.GradScaler()

    val_rec_mean_list = list()
    for epoch in range(epochs):

        net.train()
        epoch_loss = 0

        val_rec_list = list()
        with tqdm(total=len(x_train_), desc=f'Epoch {epoch + 1}/{epochs}', unit='smaple') as pbar:
            for x, y in train_loader:

                # ----train data to cuda-----
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.long)

                pred = net(x)
                weight = torch.as_tensor([0.7716, 1.5361, 0.9497]).to(device)
                # ---------OneHot_loss------------
                log_prob = torch.log_softmax(pred, dim=1)
                log_prob = log_prob * weight
                loss = -torch.sum(log_prob * y) / batch_size

                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(),  0.1)
                optimizer.step()

                # -----Mixed Precision-----
                # with amp.autocast():          ###
                #     pred = net(x)
                #     criterion = nn.BCEWithLogitsLoss()
                #     loss = criterion(pred, y)
                #
                # # epoch_loss += loss.item()
                # # optimizer.zero_grad()
                # # loss.backward()
                # # optimizer.step()
                # scaler.scale(loss).backward() ###
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                # scaler.step(optimizer)        ###
                # scaler.update()               ###
                # ----------------------
                pbar.update(x.shape[0])
                global_step += 1

                schedule.step()

                # ------REC------
                if global_step % (len(x_train_) // (3 * batch_size)) == 0:

                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    # ----check_results----
                    net.eval()
                    with torch.no_grad():
                        # ------val_set-------
                        x_val_ts = x_val_ts.to(device=device, dtype=torch.float32)
                        val_pred = net(x_val_ts)
                        # --------
                        val_log_prob = torch.log_softmax(val_pred, dim=1)
                        val_loss = -torch.sum(val_log_prob * y_val_ts) / len(y_val)
                        _, val_prelabel = torch.max(val_pred.data, 1)
                        val_prelabel = val_prelabel.data.cpu().numpy()

                        _, y_val_label = torch.max(y_val, 1)

                        val_acc = metrics.accuracy_score(y_true=y_val_label, y_pred=val_prelabel)
                        val_f1 = metrics.f1_score(y_true=y_val_label, y_pred=val_prelabel, average='macro')
                        val_precision = metrics.precision_score(y_true=y_val_label, y_pred=val_prelabel, average='macro')
                        val_recall = metrics.recall_score(y_true=y_val_label, y_pred=val_prelabel, average='macro')
                        # ---------
                        val_rec_list.append(val_loss.item())

                        # -------test-set-------
                        x_test_ts = x_test_ts.to(device=device, dtype=torch.float32)
                        test_pred = net(x_test_ts)
                        test_log_prob = torch.log_softmax(test_pred, dim=1)
                        test_loss = -torch.sum(test_log_prob * y_test_ts) / len(y_test)
                        _, test_prelabel = torch.max(test_pred.data, 1)
                        test_prelabel = test_prelabel.data.cpu().numpy()

                        _, y_test_label = torch.max(y_test, 1)

                        test_acc = metrics.accuracy_score(y_true=y_test_label, y_pred=test_prelabel)
                        test_f1 = metrics.f1_score(y_true=y_test_label, y_pred=test_prelabel, average='macro')
                        test_precision = metrics.precision_score(y_true=y_test_label, y_pred=test_prelabel, average='macro')
                        test_recall = metrics.recall_score(y_true=y_test_label, y_pred=test_prelabel, average='macro')

                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info(f'| Loss:{loss.item():.4f}')
                    logging.info(f'| Val Loss:{val_loss.item():.4f} ACC:{val_acc:.4f} F1:{val_f1:.4f} Precision:{val_precision:.4f} Recall:{val_recall:.4f}')

                    writer.add_scalar('Val', val_acc, global_step)
                    val_writer_auc.add_scalar('Val', val_f1, global_step)
                    val_writer_sensi.add_scalar('Val', val_precision, global_step)
                    val_writer_speci.add_scalar('Val', val_recall, global_step)

                    logging.info(f'| Test Loss:{test_loss.item():.4f} ACC:{test_acc:.4f} F1:{test_f1:.4f} Precision:{test_precision:.4f} Recall:{test_recall:.4f}')
                    writer.add_scalar('Test', test_acc, global_step)
                    writer_auc.add_scalar('Test', test_f1, global_step)
                    writer_sensi.add_scalar('Test', test_precision, global_step)
                    writer_speci.add_scalar('Test', test_recall, global_step)

        val_rec_mean_list.append(np.mean(val_rec_list))

        timecost = (time.time() - start)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass

            torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')

            logging.info(f'Checkpoint {epoch + 1} saved ! ')
            logging.info('TIME:%s | TIME COST:%.3f h.' % (time.ctime(), timecost / 3600))
            logging.info('--' * 70)

    logging.info(f'Best parameter on validation set: Check point {np.argmin(val_rec_mean_list)+1}')

    writer.close()
    val_writer_auc.close()
    val_writer_sensi.close()
    val_writer_speci.close()

    writer_auc.close()
    writer_sensi.close()
    writer_speci.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda')


    start = time.time()

    excel_path = os.path.abspath('../data_processed.xlsx')
    train_data = pd.read_excel(excel_path, sheet_name='train_set_p1', dtype=float)
    test_data = pd.read_excel(excel_path, sheet_name='test_set_p1', dtype=float)

    net = MLP_net_thri(num_inputs=33, n_outputs=3)
    net.to(device=device)

    logging.info(f'Network:\n'
                 f'\t{net.num_inputs} input features\n'
                 f'\t{net.num_outputs} output neurons(classes)\n')

    train_fc_net(net=net,
                device=device,
                train=train_data,
                test=test_data,
                label='three_clf',
                epochs=300,
                batch_size=128,
                lr_init=(0.0004/5),
                save_cp=True)





