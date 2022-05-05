import torch
import torch.nn.functional as F

class MLP_net_reg(torch.nn.Module):
    def __init__(self, num_inputs, n_outputs, dropout=0.5):
        super(MLP_net_reg, self).__init__()
        self.num_inputs = num_inputs
        # self.n_hiddens = n_hiddens
        self.num_outputs = n_outputs
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_0 = torch.nn.Linear(num_inputs, 1984)      # 30
        self.bn0 = torch.nn.BatchNorm1d(1984)

        self.hidden_1 = torch.nn.Linear(1984, 1408)           # 21
        self.bn1 = torch.nn.BatchNorm1d(1408)

        self.hidden_2 = torch.nn.Linear(1408, 448)             # 6
        self.bn2 = torch.nn.BatchNorm1d(448)

        self.hidden_3 = torch.nn.Linear(448, 1856)             # 1856
        self.bn3 = torch.nn.BatchNorm1d(1856)

        self.hidden_4 = torch.nn.Linear(1856, 1280)            # 19
        self.bn4 = torch.nn.BatchNorm1d(1280)

        self.hidden_5 = torch.nn.Linear(1280, 1344)             # 20
        self.bn5 = torch.nn.BatchNorm1d(1344)

        self.out = torch.nn.Linear(1344, n_outputs)

    def forward(self, x):
        x = F.leaky_relu(self.hidden_0(x))  # activation function for hidden layer
        x = self.dropout(self.bn0(x))

        x = F.leaky_relu(self.hidden_1(x))  # activation function for hidden layer
        x = self.dropout(self.bn1(x))

        x = F.leaky_relu(self.hidden_2(x))  # activation function for hidden layer
        x = self.dropout(self.bn2(x))

        x = F.leaky_relu(self.hidden_3(x))  # activation function for hidden layer
        x = self.dropout(self.bn3(x))

        x = F.leaky_relu(self.hidden_4(x))  # activation function for hidden layer
        x = self.dropout(self.bn4(x))

        x = F.leaky_relu(self.hidden_5(x))  # activation function for hidden layer
        x = self.dropout(self.bn5(x))
        x = self.out(x)
        return x