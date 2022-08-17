'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
from models import *
import numpy as np
import copy
import time
import sys

import torchvision
import torchvision.transforms as transforms

from .measure_cross_class_distances import get_train_cats


net = VGG('VGG19')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

net.load_state_dict(torch.load(
    'results/model_vgg_sgd_alpha_'+str(0.001)+'.pyc'))
# print(net)
# sys.exit(1)


def model_changed(net):
    net_new = copy.deepcopy(net)
    with torch.no_grad():
        #looping over all parameters
        for name, param in net_new.named_parameters():
            # print(name, param.size())
            # print('torch.max(param).shape=', torch.max(param).shape)
            scale = (torch.max(param) - torch.min(param)) * 0.1
            noise = scale.to(device) * \
                (torch.rand(param.size()).to(device) - 0.5) * 2.
            param.add_(noise)
    return net_new

#test Time
# t0 = time.time()
# for i in range(10):
#     net_new = model_changed(net)
# print('spent time:', time.time()-t0)


def show_value(net):
    for name, param in net.named_parameters():
        # print(name, param.size())
        if len(param.size()) == 1:
            print('sample value in the beginning')
            print(param[0])
            break
#this shows the model params indeed changed
# show_value(net)
# net_new = model_changed(net)
# show_value(net_new)

#CoriGrad: I can get the mild and dramatic directions for (1)all objects (2)each class: such as CAT
#suppose I can get different mild and dramatic directions for different classes. What do they relate to each other?


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

class_loader = get_train_cats(trainset, batch_size=1024, label=3)  # get cats
