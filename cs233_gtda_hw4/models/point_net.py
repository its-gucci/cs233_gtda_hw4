"""
Point-Net.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, init_feat_dim, conv_dims=[32, 64, 64, 128, 128], student_optional_hyper_param=None):
        """
        Students:
        You can make a generic function that instantiates a point-net with arbitrary hyper-parameters,
        or go for an implemetnations working only with the hyper-params of the HW.
        Do not use batch-norm, drop-out and other not requested features.
        Just nn.Linear/Conv1D/ reLUs and the (max) poolings.
        """
        super(PointNet, self).__init__()
        modules = []
        modules.append(nn.Conv1d(in_channels=init_feat_dim, out_channels=conv_dims[0], kernel_size=1))
        modules.append(nn.ReLU())
        for i in range(1, len(conv_dims)):
            modules.append(nn.Conv1d(in_channels=conv_dims[i-1], out_channels=conv_dims[i], kernel_size=1))
            modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(1024))
        self.pointnet = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.pointnet(x)
            
