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

from torchvision.datasets import MNIST, CIFAR10

import argparse
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from utils import progress_bar
from torch.nn.utils import parameters_to_vector as to_vector
from measure_cross_class_distances import get_train_cats
from model_gradient import concat_param_grad, aver_grad_1D
from main import classes

device = 'cuda'

def save_objects_of_class(data_loader, label):
    print(data_loader)
    total = 0
    total_objects_cls = 0
    _X = None
    _Y = None
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        index = (targets == label).nonzero()[:, 0]
        inputs_cls = inputs[index, :, :]
        targets_cls = targets[index]
        total_objects_cls += targets_cls.size(0)
        total += 1

        if _X is None:
            _X = inputs_cls
            _Y = targets_cls
        else:
            _X = torch.cat((_X, inputs_cls), dim=0)
            _Y = torch.cat((_Y, targets_cls), dim=0)

    print('total cats is ', total_objects_cls)
    print('_X.shape=', _X.shape)
    print('_Y.shape=', _Y.shape)

    torch.save(_X, 'data/cifar-10/class_'+str(label)+'_X.pt')
    torch.save(_Y, 'data/cifar-10/class_' + str(label) + '_Y.pt')

def save_objects_all_classes(data_loader):
    for cl in range(len(classes)):
        save_objects_of_class(data_loader, label=cl)

def dataset_class(label):
    _X = torch.load('data/cifar-10/class_'+str(label)+'_X.pt')
    _Y = torch.load('data/cifar-10/class_'+str(label)+'_Y.pt')
    dataset = TensorDataset(_X, _Y)
    return dataset

def train_loader_class(label):
    dataset = dataset_class(label)
    return torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True, num_workers=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    print('@@lr=', args.lr)

    net = VGG('VGG19')
    # net = net.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        # trainset = torchvision.datasets.CIFAR10(
        #     root='./data', train=True, download=True, transform=transform_train)

        #It appears data transform is applied in dataloader, not in the CIFAR: transform has no effect here yet
        # trainset = CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        #However, on MNIST, the data transform is effective in the dataset already.
        # trainset = MNIST('./data', transform=transforms.ToTensor(), download=True)

        #so here I used a solution that first retrieve from the dataloader, save, and then load; this guarantees using the same transformed data as trainloader

        trainset = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4096, shuffle=True, num_workers=2)

        #this is just one time running.
        # save_objects_all_classes(trainloader)
        # sys.exit(1)

        # train_cls = dataset_cls(label=3)
        # train_cls_loader = torch.utils.data.DataLoader(train_cls, batch_size=1000, shuffle=True, num_workers=2)
        #
        # for batch_idx, (inputs, targets) in enumerate(train_cls_loader):
        #     print('targets=', targets)
        #     print('inputs.shape=', inputs.shape)

        net = torch.nn.DataParallel(net)

        # model_path = 'results/model_vgg_sgd_alpha_'+str(0.1)
        model_path = 'results/model_vgg_sgd_alpha_'+str(0.01)
        # model_path = 'results/model_vgg_sgd_alpha_'+str(0.001)
        # model_path = 'results/model_vgg_sgd_alpha_'+str(0.001)+'_batchsize1024'
        print('loading model at path:', model_path)
        net.load_state_dict(torch.load(model_path+'.pyc'))
        # print(net)

        criterion = nn.CrossEntropyLoss(reduction='sum')  # by default. it's mean.
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0, weight_decay=0)  # vannila SGD

        #This computes the gradient for each class
        grad_cls_norm2 = np.zeros(len(classes))
        for cl in range(len(classes)):
            print('class: ', classes[cl])
            trainloader_cls = train_loader_class(label=cl)

            grad = aver_grad_1D(trainloader_cls, net, optimizer, criterion)
            print('grads.shape=', grad.shape)
            np.save(model_path+'_grad_'+classes[cl]+'.npy', grad)

            grad_cls_norm2[cl] = np.linalg.norm(grad)

        np.save(model_path+'_grad_norm2_classes.npy', grad_cls_norm2)