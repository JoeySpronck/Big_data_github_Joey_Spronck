#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch, torchvision
from torch import nn
import torch.nn.functional as F
# from torch import optim
# from torch.autograd import Variable

# import numpy as np
# import matplotlib.pyplot as plt

# torch.set_printoptions(linewidth=120)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)
        
        self.drop = nn.Dropout()
        
        self.fc1 = nn.Linear(in_features = 12*4*4, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)
        
    def forward(self, t):
        t = F.relu(self.conv1(t)) # HIDDEN CONV LAYER 1
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        
        t = F.relu(self.conv2(t)) # HIDDEN CONV LAYER 2
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        
        t = F.relu(self.fc1(self.drop(t.reshape(-1, 12*4*4)))) # HIDDEN FULLY CONNECTED LAYER 1
        t = F.relu(self.fc2(self.drop(t))) # HIDDEN FULLY CONNECTED LAYER 2
        t = self.out(t) # OUTPUT LAYER
        
        return t 

