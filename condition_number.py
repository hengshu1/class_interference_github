'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import numpy as np
import copy
import time

net = VGG('VGG19')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

net.load_state_dict('results/model_vgg_sgd_alpha_'+str(args.lr)+'.pyc')

# I should have a deep copy of the net before hand


def model_changed(net):
    net_new = copy.deepcopy(net)
    #looping over all parameters
    for name, param in net_new.named_parameters():
        print(name, param.size())

        print('torch.max(param).shape=', torch.max(param).shape)
        scale = (torch.max(param) - torch.min(param)) * 0.1
        noise = scale * (torch.rand(param.size()) - 0.5) * 2.
        param.add_(noise)
    return net_new

#test Time
t0 = time.time()
for i in range(10):
    net_new = model_changed(net)
print('spent time:', time.time()-t0)
