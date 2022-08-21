
import torch
import torch.nn as nn
from models import *
import torch.optim as optim
import numpy as np
import copy
import time
import sys
import torch.backends.cudnn as cudnn

import argparse
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar
from torch.nn.utils import parameters_to_vector as to_vector
# from .measure_cross_class_distances import get_train_cats
from main import device
from torchvision.datasets import MNIST, CIFAR10


def test(trainloader, net, optimizer, criterion):
    # net.train()
    net.eval()

    train_loss = 0
    correct = 0
    total = 0

    grads_sum = {}
    for name, param in net.named_parameters():
        grads_sum[name] = None
    # print('grads_sum=', grads_sum)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # optimizer.step()

        # print('loss=', loss.item())

        if batch_idx == 0:
            print('targets=', targets)
            # print('inputs.shape=', inputs.shape)
            print('inputs=', inputs[:, 0, 0, 0])
            for name, param in net.named_parameters():
                # print('name=', name)
                print('@@@@GRAD')
                print(param.grad[:7, 0, 0, 0])
                return param.grad

    return

#torch.random.manual_seed(1)
torch.manual_seed(1)
import random
random.seed(1)

net = VGG('VGG19')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

# Data
print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465),
#                          (0.2023, 0.1994, 0.2010)),
# ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# trainset = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
trainset = CIFAR10(root='./data', train=True, transform=transform_test, download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss(reduction='sum')  # by default. it's mean.
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0, weight_decay=0)  # vannila SGD

for name, param in net.named_parameters():
    print('initial w=', name)
    print(param[:7, 0, 0, 0])
    break

#gradient is different every time I run it: I think this is due to the deep backprogagation is a long chain and numerical senstive.
# I just couldn't understand the output and error is the same (maybe not?) Loss is the same. checked.
# is it numerical difference?
test(trainloader, net, optimizer, criterion)

for name, param in net.named_parameters():
    print('after test w=', name)
    print(param[:7, 0, 0, 0])
    break

#second run: different gradient for the same data: something must be wrong.
#just found the input is changed!! It's because the transform has randomCrop and random flip.
#In PyTorch 1.6, you need to use torch.manual_seed(5) and random.seed(5) same time. Please see the issue in: https://github.com/pytorch/pytorch/issues/42331 169
#changed to test transform: no randomness: still grad is a bit different
#checked w didn't change after the first test

print('---------second run-----')
test(trainloader, net, optimizer, criterion)

'''
Found out: it's the BN layer. should use net.eval()
Why is there such a big difference. Most gradients are now near 0. 
'''