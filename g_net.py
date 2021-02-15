import torch
import torch.nn as nn
import torch.nn.functional as Func

class G_net_dnn (nn.Module):
    def __init__ (self, embedding_dim):
        super(G_net_dnn, self).__init__()
        self.linear1a = nn.Linear(embedding_dim, 32)
        self.linear1b = nn.Linear(embedding_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.linear2 = nn.Linear(32, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.linear3 = nn.Linear(32, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.linear4 = nn.Linear(32, embedding_dim)

    def forward (self, X1, X2):
        linear1 = Func.relu(self.bn1(self.linear1a(X1) + self.linear1a(X2) + self.linear1b(X1*X2)))
        linear2 = Func.relu(self.bn2(self.linear2(linear1)))
        linear3 = Func.relu(self.bn3(self.linear3(linear2)))
        linear4 = Func.normalize(self.linear4(linear3))
        return linear4

class G_net_linear_fc (nn.Module):
    def __init__ (self, embedding_dim):
        super(G_net_linear_fc, self).__init__()
        self.linear1a = nn.Linear(embedding_dim, 32)
        self.linear1b = nn.Linear(embedding_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.linear4 = nn.Linear(32, embedding_dim)

    def forward (self, X1, X2):
        linear1 = Func.relu(self.bn1(self.linear1a(X1) + self.linear1a(X2) + self.linear1b(X1*X2)))
        linear4 = Func.normalize(self.linear4(linear1))
        return linear4

class G_net_linear (nn.Module):
    def __init__ (self, embedding_dim):
        super(G_net_linear, self).__init__()
        self.linear1a = nn.Linear(embedding_dim, 32)
        self.linear1b = nn.Linear(embedding_dim, 32)

    def forward (self, X1, X2):
        linear4 = Func.normalize(self.linear1a(X1) + self.linear1a(X2) + self.linear1b(X1*X2))
        return linear4

class G_net_mean (nn.Module):
    def __init__ (self):
        super(G_net_mean, self).__init__()

    def forward (self, X1, X2):
        linear4 = Func.normalize(X1+X2)
        return linear4