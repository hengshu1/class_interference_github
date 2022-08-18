'''Train CIFAR10 with PyTorch.'''
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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--batchsize', default=128, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

print('@@lr=', args.lr)
print('@@batchsize=', args.batchsize)

net = VGG('VGG19')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# model_path = 'results/model_vgg_sgd_alpha_'+str(0.001)
model_path = 'results/model_vgg_sgd_alpha_'+str(0.001)+'_batchsize1024'
net.load_state_dict(torch.load(model_path+'.pyc'))
# print(net)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

criterion = nn.CrossEntropyLoss(reduction='sum')  # by default. it's mean.
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0, weight_decay=0)  # first do without momentum

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)


def check_param_grad(net):
    '''
    fix x = net; we are interested at the final model centered. compute f'(x)
    And check the curvatures around it; and see any direction going down? any leaky direction? If mu<0, then it's leaky
    '''
    first_time = True
    for name, param in net.named_parameters():
        # print(name, param.size())
        # print(name, 'grad size:', param.grad.size())
        _wgrad = to_vector(param.grad)
        if first_time:
            wg_all = _wgrad
            first_time = False
        # print('1d len:', _wgrad.size())
        else:
            wg_all = torch.cat((wg_all, _wgrad))

    print('total params from w_all=', wg_all.size())
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print('total params =', pytorch_total_params)  # same
    return wg_all


def aver_grad(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    first_time = True
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        grads = check_param_grad(net)
        if first_time:
            grads_sum = grads
            first_time = False
        else:
            grads_sum += grads


    grads_sum /= total
    print('total samples=', total)
    return grads_sum.cpu().numpy()


#only one epoch
grad = train(1)
print('grads.shape=', grad.shape)
np.save(model_path+'_grad.npy', grad)

# class_loader = get_train_cats(trainset, batch_size=1024, label=3)  # get cats
#how many cats?
