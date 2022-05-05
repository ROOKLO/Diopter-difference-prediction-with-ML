import os
import logging
import os
import time

import numpy as np
import pandas as pd
from ranger import Ranger
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.cuda import amp
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Focal_Loss import *
from MLP_net import *

dir_checkpoint = 'checkpoints/'


def train_fc_net(net, device, train, test, label, epochs, batch_size, lr_init, save_cp):
    """
    输入的train, test为dataframe格式
    """
    x_train, y_train = train.iloc[:, :33], train[label]

    x_test, y_test = test.iloc[:, :33], test[label]
    # ----test_set-------
    scaler_2 = StandardScaler().fit(x_train)
    x_test_scaled = scaler_2.transform(x_test)
    x_test_ts = torch.as_tensor(x_test_scaled)

    # ----train_set------
    x_train_, x_val, y_train_, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=7)

    scaler_1 = StandardScaler().fit(x_train_)
    x_train_scaled = scaler_1.transform(x_train_)
    x_val_scaled = scaler_1.transform(x_val)

    x_train_ts = torch.as_tensor(x_train_scaled)
    y_train_ts = torch.as_tensor(y_train_.values)
    train_dataset = TensorDataset(x_train_ts, y_train_ts)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # ------val_set------
    x_val_ts = torch.as_tensor(x_val_scaled)
    y_val = torch.as_tensor(y_val.values.reshape(-1, 1))
    y_val_ts = y_val.to(device)


    writer = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}_{label}/acc')
    val_writer_auc = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}_{label}/val_auc')
    val_writer_sensi = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}_{label}/val_sensitivity')
    val_writer_speci = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}_{label}/val_specificity')

    writer_auc = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}_{label}/test_auc')
    writer_sensi = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}_{label}/test_sensiticity')
    writer_speci = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}_{label}/test_specificity')

    global_step = 0                             # 1batch = 1step
    logging.info(f'''Starting training:
                    Epochs:         {epochs}
                    Batch_size:     {batch_size}
                    learning rate:  {lr_init}
                    Training size:  {int(len(train)*0.9)}
                    Validation size:{int(len(train)*0.1)}
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
                # for x, y in zip(batch[0], batch[1].view(-1, 1)):
                # x = batch[0]
                # y = batch[1]
                # ----train data to cuda-----
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32).view(-1, 1)

                pred = net(x)
                # -----BCEWithLogitsLoss-----
                posi_weight_25 = torch.as_tensor([0.756])
                posi_weight_5 = torch.as_tensor([1.071])
                posi_weight_75 = torch.as_tensor([1.640])

                criterion = nn.BCEWithLogitsLoss(pos_weight=posi_weight_75).to(device)
                loss = criterion(pred, y)

                # ----focal_loss----
                # criterion = FocalLoss()
                # loss = criterion(pred, y, class_weight=(0.2, 0.8), type='sigmoid')
                # # 每个batch的loss累加每次epoch清零
                # writer.add_scalar('Loss/train', loss.item(), global_step)
                # pbar.set_postfix(**{'loss (batch)': loss.item()})

                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(),  0.1)
                # # 梯度裁剪 Clips gradient of an iterable of parameters at specified value.
                optimizer.step()

                # -----混合精度运算-----
                # with amp.autocast():          ###
                #     pred = net(x)
                #     criterion = nn.BCEWithLogitsLoss()
                #     loss = criterion(pred, y)

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

                # ------记录------
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
                        criterion_2 = nn.BCEWithLogitsLoss()
                        val_loss = criterion_2(val_pred, y_val_ts)

                        val_sig = torch.sigmoid(val_pred).data.cpu().numpy()
                        val_pre_label = (val_sig > 0.5).reshape(-1)
                        val_acc = metrics.accuracy_score(y_val, val_pre_label)
                        val_auc = metrics.roc_auc_score(y_val, val_sig.reshape(-1))
                        tn, fp, fn, tp = metrics.confusion_matrix(y_val, val_pre_label).ravel()
                        val_sensitivity = tp / (tp + fn)
                        val_specificity = tn / (tn + fp)

                        # acc_balanced = metrics.balanced_accuracy_score(y_val, val_pre_label)
                        val_rec_list.append(val_loss.item())

                        # -------test-set-------
                        x_test_ts = x_test_ts.to(device=device, dtype=torch.float32)
                        test_pred = net(x_test_ts)
                        test_sig = torch.sigmoid(test_pred).data.cpu().numpy()
                        test_pre_label = (test_sig > 0.5).reshape(-1)
                        test_acc = metrics.accuracy_score(y_test, test_pre_label)
                        test_auc = metrics.roc_auc_score(y_test, test_sig.reshape(-1))
                        tn, fp, fn, tp = metrics.confusion_matrix(y_test, test_pre_label).ravel()
                        test_sensitivity = tp/(tp+fn)
                        test_specificity = tn/(tn+fp)

                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info(f'| Loss:{loss.item():.4f}')
                    logging.info(f'| Val Loss:{val_loss.item():.4f} ACC:{val_acc:.4f} AUC:{val_auc:.4f} sensi:{val_sensitivity:.4f} speci:{val_specificity:.4f}')

                    writer.add_scalar('Val', val_acc, global_step)
                    val_writer_auc.add_scalar('Val', val_auc, global_step)
                    val_writer_sensi.add_scalar('Val', val_sensitivity, global_step)
                    val_writer_speci.add_scalar('Val', val_specificity, global_step)

                    logging.info(f'| Test ACC:{test_acc:.4f} AUC:{test_auc:.4f} sensi:{test_sensitivity:.4f} speci:{test_specificity:.4f}')
                    writer.add_scalar('Test', test_acc, global_step)
                    writer_auc.add_scalar('Test', test_auc, global_step)
                    writer_sensi.add_scalar('Test', test_sensitivity, global_step)
                    writer_speci.add_scalar('Test', test_specificity, global_step)

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

#----模型参数加载---
    start = time.time()

    excel_path = os.path.abspath('data_processed.xlsx')
    train_data = pd.read_excel(excel_path, sheet_name='train_set', dtype=float)
    test_data = pd.read_excel(excel_path, sheet_name='test_set', dtype=float)

    net = MLP_net_25(num_inputs=33, n_outputs=1)
    net.to(device=device)

    logging.info(f'Network:\n'
                 f'\t{net.num_inputs} input features\n'
                 f'\t{net.num_outputs} output neurons(classes)\n')

    # train_fc_net(net=net,
    #             device=device,
    #             train=train_data,
    #             test=test_data,
    #             label='label0.25',
    #             epochs=50,
    #             batch_size=256,
    #             lr_init=(0.0003/5),
    #             save_cp=True)

    # train_fc_net(net=net,
    #             device=device,
    #             train=train_data,
    #             test=test_data,
    #             label='label0.5',
    #             epochs=50,
    #             batch_size=256,
    #             lr_init=(0.0001/5),
    #             save_cp=True)

    train_fc_net(net=net,
                device=device,
                train=train_data,
                test=test_data,
                label='label0.75',
                epochs=50,
                batch_size=256,
                lr_init=(0.0005/5),
                save_cp=True)








