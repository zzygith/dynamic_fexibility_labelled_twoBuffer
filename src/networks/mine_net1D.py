import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class MINE_net1D(BaseNet):

    def __init__(self):
        super().__init__()

        #2d: 15
        #20: 50
        rep = 15
        output_dim=2
        self.rep_dim = output_dim
        

        self.fc1 = nn.Linear(1, rep, bias=False)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(rep, rep, bias=False)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(rep, output_dim, bias=False)
        
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
