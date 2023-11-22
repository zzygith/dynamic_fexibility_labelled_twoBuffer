import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class MINE_nettwoBuffer2D(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 18
        #self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 4, 3, stride=1, padding=1)
        #self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(4, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 4, 3, stride=1, padding=1)
        #self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 2 * 6, self.rep_dim)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        #x = self.pool(F.leaky_relu(x))
        x = F.tanh(self.conv2(x))
        #x = self.pool(F.leaky_relu(x))
        x = F.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


