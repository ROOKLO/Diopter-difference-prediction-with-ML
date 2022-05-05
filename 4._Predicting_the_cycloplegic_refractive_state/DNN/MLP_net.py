import torch
import torch.nn.functional as F


class MLP_net_thri(torch.nn.Module):
    def __init__(self, num_inputs, n_outputs, dropout=0.280):
        super(MLP_net_thri, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = n_outputs
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_0 = torch.nn.Linear(num_inputs, (20+1)*32)
        self.bn0 = torch.nn.BatchNorm1d((20+1)*32)

        self.hidden_1 = torch.nn.Linear((20+1)*32, (8+1)*32)
        self.bn1 = torch.nn.BatchNorm1d((8+1)*32)

        self.hidden_2 = torch.nn.Linear((8+1)*32, (3+1)*32)
        self.bn2 = torch.nn.BatchNorm1d((3+1)*32)

        self.hidden_3 = torch.nn.Linear((3+1)*32, (15+1)*32)
        self.bn3 = torch.nn.BatchNorm1d((15+1)*32)

        self.hidden_4 = torch.nn.Linear((15+1)*32, (2+1)*32)
        self.bn4 = torch.nn.BatchNorm1d((2+1)*32)

        self.out = torch.nn.Linear((2+1)*32, n_outputs)

    def forward(self, x):
        x = F.relu(self.hidden_0(x))
        x = self.dropout(self.bn0(x))

        x = F.relu(self.hidden_1(x))
        x = self.dropout(self.bn1(x))

        x = F.relu(self.hidden_2(x))
        x = self.dropout(self.bn2(x))

        x = F.relu(self.hidden_3(x))
        x = self.dropout(self.bn3(x))

        x = F.relu(self.hidden_4(x))
        x = self.dropout(self.bn4(x))

        x = self.out(x)
        return x