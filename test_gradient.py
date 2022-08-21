
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

        if batch_idx == 0:
            print('targets=', targets)
            for name, param in net.named_parameters():
                print('name=', name)
                print(param.grad[:3, 0, 0, 0])
                return param.grad

    return

net = VGG('VGG19')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

trainset = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4096, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss(reduction='sum')  # by default. it's mean.
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0, weight_decay=0)  # vannila SGD


test(trainloader, net, optimizer, criterion)

#second run: same gradient?
test(trainloader, net, optimizer, criterion)