import torch
import torch.nn.functional as F

class MLP_net_25(torch.nn.Module):
    def __init__(self, num_inputs, n_outputs, dropout=0.46):
        super(MLP_net_25, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = n_outputs
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_0 = torch.nn.Linear(num_inputs, (26+1)*32)
        self.bn0 = torch.nn.BatchNorm1d((26+1)*32)

        self.hidden_1 = torch.nn.Linear((26+1)*32, (5+1)*32)
        self.bn1 = torch.nn.BatchNorm1d((5+1)*32)

        self.hidden_2 = torch.nn.Linear((5+1)*32, (15+1)*32)  # hidden layer
        self.bn2 = torch.nn.BatchNorm1d((15+1)*32)

        self.hidden_3 = torch.nn.Linear((15+1)*32, (2+1)*32)
        self.bn3 = torch.nn.BatchNorm1d((2+1)*32)

        self.hidden_4 = torch.nn.Linear((2+1)*32, (4+1)*32)  # hidden layer
        self.bn4 = torch.nn.BatchNorm1d((4+1)*32)

        self.out = torch.nn.Linear((4+1)*32, n_outputs)  # output layer

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

        x = self.out(x)
        return x


class MLP_net_5(torch.nn.Module):
    def __init__(self, num_inputs, n_outputs, dropout=0.467):
        super(MLP_net_5, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = n_outputs
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_0 = torch.nn.Linear(num_inputs, (19+1)*32)
        self.bn0 = torch.nn.BatchNorm1d((19+1)*32)

        self.hidden_1 = torch.nn.Linear((19+1)*32, (3+1)*32)
        self.bn1 = torch.nn.BatchNorm1d((3+1)*32)

        self.hidden_2 = torch.nn.Linear((3+1)*32, (10+1)*32)  # hidden layer
        self.bn2 = torch.nn.BatchNorm1d((10+1)*32)

        self.hidden_3 = torch.nn.Linear((10+1)*32, (0+1)*32)
        self.bn3 = torch.nn.BatchNorm1d((0+1)*32)

        self.hidden_4 = torch.nn.Linear((0+1)*32, (10+1)*32)  # hidden layer
        self.bn4 = torch.nn.BatchNorm1d((10+1)*32)

        self.out = torch.nn.Linear((10+1)*32, n_outputs)  # output layer

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

        x = self.out(x)
        return x


class MLP_net_75(torch.nn.Module):
    def __init__(self, num_inputs, n_outputs, dropout=0.486):
        super(MLP_net_75, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = n_outputs
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_0 = torch.nn.Linear(num_inputs, (22+1)*32)
        self.bn0 = torch.nn.BatchNorm1d((22+1)*32)

        self.hidden_1 = torch.nn.Linear((22+1)*32, (1+1)*32)
        self.bn1 = torch.nn.BatchNorm1d((1+1)*32)

        self.hidden_2 = torch.nn.Linear((1+1)*32, (5+1)*32)  # hidden layer
        self.bn2 = torch.nn.BatchNorm1d((5+1)*32)

        self.hidden_3 = torch.nn.Linear((5+1)*32, (8+1)*32)
        self.bn3 = torch.nn.BatchNorm1d((8+1)*32)

        self.hidden_4 = torch.nn.Linear((8+1)*32, (4+1)*32)  # hidden layer
        self.bn4 = torch.nn.BatchNorm1d((4+1)*32)

        self.out = torch.nn.Linear((4+1)*32, n_outputs)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden_0(x))  # activation function for hidden layer
        x = self.dropout(self.bn0(x))

        x = F.relu(self.hidden_1(x))  # activation function for hidden layer
        x = self.dropout(self.bn1(x))

        x = F.relu(self.hidden_2(x))  # activation function for hidden layer
        x = self.dropout(self.bn2(x))

        x = F.relu(self.hidden_3(x))  # activation function for hidden layer
        x = self.dropout(self.bn3(x))

        x = F.relu(self.hidden_4(x))  # activation function for hidden layer
        x = self.dropout(self.bn4(x))

        x = self.out(x)
        return x


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

        self.hidden_2 = torch.nn.Linear((8+1)*32, (3+1)*32)  # hidden layer
        self.bn2 = torch.nn.BatchNorm1d((3+1)*32)

        self.hidden_3 = torch.nn.Linear((3+1)*32, (15+1)*32)
        self.bn3 = torch.nn.BatchNorm1d((15+1)*32)

        self.hidden_4 = torch.nn.Linear((15+1)*32, (2+1)*32)  # hidden layer
        self.bn4 = torch.nn.BatchNorm1d((2+1)*32)

        self.out = torch.nn.Linear((2+1)*32, n_outputs)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden_0(x))  # activation function for hidden layer
        x = self.dropout(self.bn0(x))

        x = F.relu(self.hidden_1(x))  # activation function for hidden layer
        x = self.dropout(self.bn1(x))

        x = F.relu(self.hidden_2(x))  # activation function for hidden layer
        x = self.dropout(self.bn2(x))

        x = F.relu(self.hidden_3(x))  # activation function for hidden layer
        x = self.dropout(self.bn3(x))

        x = F.relu(self.hidden_4(x))  # activation function for hidden layer
        x = self.dropout(self.bn4(x))

        x = self.out(x)
        return x

if __name__ == '__main__':
    print(MLP_net_25(num_inputs=33, n_outputs=1))