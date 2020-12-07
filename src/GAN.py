import torch
import torch.nn as nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        self.map3 = nn.Linear(hidden_size, hidden_size)
        # self.bn3 = nn.BatchNorm1d(hidden_size)
        self.map4 = nn.Linear(hidden_size, output_size)
        self.lrelu = nn.LeakyReLU(negative_slope=0.02)
        # self.relu=nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        # self.f = fd

    def forward(self, x):
        x = self.lrelu(self.map1(x))
        x = self.lrelu(self.map2(x))
        x = self.lrelu(self.map3(x))
        return self.Sigmoid(self.map4(x))
