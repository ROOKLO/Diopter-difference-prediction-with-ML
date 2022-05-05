import logging
import logging
import os
import time

import numpy as np
import pandas as pd
import torch.nn as nn
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from MLP_net import *

dir_checkpoint = 'checkpoints/'

def train_mlp(net, device, train, label, epochs, batch_size, lr_init, save_cp):
    x_train, y_train = train.iloc[:, :-6], train[label]
    x_train_, x_val, y_train_, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

    scaler_1 = StandardScaler().fit(x_train_)
    x_train_scaled = scaler_1.transform(x_train_)
    x_val_scaled = scaler_1.transform(x_val)

    x_train_ts = torch.as_tensor(x_train_scaled)
    y_train_ts = torch.as_tensor(y_train_.values)
    train_dataset = TensorDataset(x_train_ts, y_train_ts)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # ------val_set------
    x_val_ts = torch.as_tensor(x_val_scaled)

    val_writer_mse = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}/mse')
    val_writer_mae = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}/mae')
    val_writer_r2 = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}/r2')
    val_writer_r = SummaryWriter(comment=f'Ranger_LR_{lr_init}_BS_{batch_size}/r')

    global_step = 0

    logging.info(f'''Starting training:
                    Epochs:         {epochs}
                    Batch_size:     {batch_size}
                    learning rate:  {lr_init}
                    Training size:  {int(len(train)*0.9)}
                    Validation size:{int(len(train)*0.1)}
                    Checkpoints:    {save_cp}
                    Device:         {device.type}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr_init, weight_decay=3e-04)

    val_rec_mean_list = list()
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        val_rec_list = list()
        with tqdm(total=len(x_train_), desc=f'Epoch {epoch + 1}/{epochs}', unit='smaple') as pbar:
            for x, y in train_loader:
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32).view(-1, 1)

                pred = net(x)
                criterion = nn.MSELoss()
                loss = criterion(pred, y)

                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(),  0.1)
                optimizer.step()

                pbar.update(x.shape[0])
                global_step += 1

                if global_step % (len(x_train_) // (2 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        val_writer_mse.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        val_writer_mse.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    net.eval()
                    with torch.no_grad():
                        x_val_ts = x_val_ts.to(device=device, dtype=torch.float32)
                        val_pred = net(x_val_ts).data.cpu().numpy().ravel()
                        val_mse = metrics.mean_squared_error(y_true=y_val, y_pred=val_pred)
                        val_mae = metrics.mean_absolute_error(y_true=y_val, y_pred=val_pred)
                        val_r2 = metrics.r2_score(y_true=y_val, y_pred=val_pred)
                        val_r, p = stats.pearsonr(x=y_val, y=val_pred)

                        val_rec_list.append(val_r2)

                    val_writer_mse.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info(f'| Loss:{loss.item():.4f}')
                    logging.info(f'| Val MSE:{val_mse:.4f} MAE:{val_mae:.4f} r2:{val_r2:.4f} r:{val_r:.4f}')


                    val_writer_mse.add_scalar('Val', val_mse, global_step)
                    val_writer_mae.add_scalar('Val', val_mae, global_step)
                    val_writer_r2.add_scalar('Val', val_r2, global_step)
                    val_writer_r.add_scalar('Val', val_r, global_step)

            val_writer_mse.add_scalar('Train', epoch_loss/len(train_loader), epoch)

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

    logging.info(f'Best parameter on validation set: Check point {np.argmax(val_rec_mean_list) + 1}')

    val_writer_mse.close()
    val_writer_mae.close()
    val_writer_r2.close()
    val_writer_r.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda')

    start = time.time()

    excel_path = os.path.abspath('../../data6.xlsx')
    train_data = pd.read_excel(excel_path, sheet_name='train_set_p1', dtype=float)

    label = 'post_SE_R'
    net = MLP_net_reg(num_inputs=33, n_outputs=1)
    net.to(device=device)

    logging.info(f'Network:\n'
                 f'\t{net.num_inputs} input features\n'
                 f'\t{net.num_outputs} output neurons(classes)\n')

    train_mlp(net=net,
              device=device,
              train=train_data,
              label='post_SE_R',
              epochs=100,
              batch_size=256,
              lr_init= 0.0006,
              save_cp=True)