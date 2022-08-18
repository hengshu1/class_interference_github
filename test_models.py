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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def test(net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


# Models
print('==> Building model..')

model_001_128 = VGG('VGG19')
model_001_128.load_state_dict(torch.load(
    'results/model_vgg_sgd_alpha_'+str(0.001)+'.pyc'))
print('model_001_128')
test(model_001_128)

model_001_1024 = VGG('VGG19')
model_001_1024.load_state_dict(torch.load(
    'results/model_vgg_sgd_alpha_'+str(0.001)+'_batchsize1024.pyc'))
print('model_001_1024')
test(model_001_1024)

model_0001_128 = VGG('VGG19')
model_0001_128.load_state_dict(torch.load(
    'results/model_vgg_sgd_alpha_'+str(0.0001)+'.pyc'))
print('model_0001_128')
test(model_0001_128)

model_0001_1024 = VGG('VGG19')
model_0001_1024.load_state_dict(torch.load(
    'results/model_vgg_sgd_alpha_'+str(0.0001)+'_batchsize1024.pyc'))
print('model_0001_1024')
test(model_0001_1024)
