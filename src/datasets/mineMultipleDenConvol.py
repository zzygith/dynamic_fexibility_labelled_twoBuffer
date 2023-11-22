from base.torchvision_dataset import TorchvisionDataset

import math
import torch
import pandas
import numpy as np

class MINEMDCONVOL_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):


        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = [0]
        self.outlier_classes = [1]


        self.train_set = []
        trdat = np.array(pandas.read_csv(root, header=None,sep='\t'))

        trdat=trdat.reshape((-1,7,2)) #2d uncertain parameter, 6 periods and 1 row for [0,0] or [1,1]

        for i in range(0,len(trdat)):
            self.train_set.append((torch.Tensor(trdat[i][:].astype(np.float)),0,i))
        
        
        self.test_set = []


    def __getitem__(self, index):
        if self.train:
            return self.train_set[index][0], self.train_set[index][1], index
        else:
            return self.test_set[index][0], self.test_set[index][1], index

