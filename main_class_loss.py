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

from ego_models import train_G_matrix
from main import classes, train

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size')
    parser.add_argument('--model', default='vgg19', type=str, help='model name')
    parser.add_argument('--lr_mode', default='constant', type=str, help='learning rate mode')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    trainloader_big = torch.utils.data.DataLoader(
        trainset, batch_size=1024, shuffle=False, num_workers=2)


    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)


    # Models
    print('==> Building model..')
    args.model=args.model.lower()
    print('running model:', args.model)
    args.lr_mode = args.lr_mode.lower()
    print('lr mode=', args.lr_mode)

    if args.model == 'vgg19':
        net = VGG('VGG19')
    elif args.model == 'resnet18':
        net = ResNet18()
    else:
        print('not run yet')
        sys.exit(1)

    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)  # first do without momentum

    if args.lr_mode == 'constant' or args.lr_mode == 'fixed':
        scheduler = None
    elif args.lr_mode == 'schedule' or args.lr_mode == 'anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        print('lr mode not supported this yet. ')


# def evaluate_f_class():
#     '''evaluate the loss for each class on the whole training dataset: no training.
#     note this one uses the original loss; and it's tricky: the loss is averaged over batch; and then summed across batches
#     '''
#     train_losses_class = np.zeros(len(classes))
#     net.eval()
#     for batch_idx, (inputs, targets) in enumerate(trainloader_big):
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = net(inputs)
#         for cl in range(len(classes)):
#             index = (targets == cl).nonzero()[:, 0]
#             loss_c = criterion(outputs[index], targets[index])
#             train_losses_class[cl] += loss_c.item()
#     return train_losses_class

    fc_loss = []
    num_epochs = 200
    for epoch in range(num_epochs):
        train(epoch, net, criterion, optimizer, trainloader)
        # test(epoch)
        # fc_loss.append(evaluate_f_class())
        fc_loss.append(train_G_matrix(net, criterion, trainloader))

        if args.lr_mode=='schedule' or args.lr_mode=='anneal':
            scheduler.step()
            if epoch % 20 == 0 or epoch == num_epochs -1:
                print('last lr=', scheduler.get_last_lr())

    fc_loss = np.array(fc_loss)
    print('fc_loss.shape', fc_loss.shape)
    # file_name = 'results/fc_' + args.model+'lrmode_'+ args.lr_mode + '_sgd_alpha_'+str(args.lr)+'.npy'
    file_name = 'results/Gmatrix_' + args.model+'lrmode_'+ args.lr_mode + '_sgd_alpha_'+str(args.lr)+'.npy'
    np.save(file_name, fc_loss)
