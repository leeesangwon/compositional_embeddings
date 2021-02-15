import torch
import torch.nn as nn
import torch.nn.functional as Func

class Mike_net_dnn(nn.Module):
    def __init__(self, embedding_dim):
        super(Mike_net_dnn, self).__init__()
        self.linear1a = nn.Linear(embedding_dim, embedding_dim)
        self.linear1b = nn.Linear(embedding_dim, embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.linear3 = nn.Linear(embedding_dim, embedding_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        self.linear4 = nn.Linear(embedding_dim, 1)

    def forward(self, X1, X2):
        linear1 = Func.relu(self.bn1(self.linear1a(X1) + self.linear1b(X2)))# + self.linear1b(X1*X2)))
        linear2 = Func.relu(self.bn2(self.linear2(linear1)))
        linear3 = Func.relu(self.bn3(self.linear3(linear2)))
        linear4 = torch.sigmoid(self.linear4(linear3))
        return linear4

class Mike_net_linear_fc(nn.Module):
    def __init__(self, embedding_dim):
        super(Mike_net_linear_fc, self).__init__()
        self.linear1a = nn.Linear(embedding_dim, embedding_dim)
        self.linear1b = nn.Linear(embedding_dim, embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.linear4 = nn.Linear(embedding_dim, 1)

    def forward(self, X1, X2):
        linear1 = Func.relu(self.bn1(self.linear1a(X1) + self.linear1b(X2)))# + self.linear1b(X1*X2)))
        linear4 = torch.sigmoid(self.linear4(linear1))
        return linear4

class Mike_net_linear(nn.Module):
    def __init__(self, embedding_dim):
        super(Mike_net_linear, self).__init__()
        self.linear4 = nn.Linear(embedding_dim*2, 1)

    def forward(self, X1, X2):
        linear4 = torch.sigmoid(self.linear4(torch.cat([X1, X2], 1)))
        return linear4
