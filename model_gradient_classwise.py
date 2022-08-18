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
from measure_cross_class_distances import get_train_cats
from model_gradient import check_param_grad, aver_grad
from main import classes, device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    print('@@lr=', args.lr)

    net = VGG('VGG19')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
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
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)

        net = torch.nn.DataParallel(net)

        # model_path = 'results/model_vgg_sgd_alpha_'+str(0.1)
        model_path = 'results/model_vgg_sgd_alpha_'+str(0.01)
        # model_path = 'results/model_vgg_sgd_alpha_'+str(0.001)
        # model_path = 'results/model_vgg_sgd_alpha_'+str(0.001)+'_batchsize1024'
        print('loading model at path:', model_path)
        net.load_state_dict(torch.load(model_path+'.pyc'))
        # print(net)

        # by default. it's mean.
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0, weight_decay=0)  # first do without momentum

        grad_cls_norm2 = np.zeros(len(classes))
        for cl in range(len(classes)):
            print('class: ', classes[cl])
            trainloader = get_train_cats(
                trainset, batch_size=1024, label=cl)  # get objects

            grad = aver_grad(trainloader, net, optimizer, criterion)
            print('grads.shape=', grad.shape)
            np.save(model_path+'_grad_'+classes[cl]+'.npy', grad)

            grad_cls_norm2[cl] = np.linalg.norm(grad)

        np.save(model_path+'_grad_norm2_classes.npy', grad_cls_norm2)
