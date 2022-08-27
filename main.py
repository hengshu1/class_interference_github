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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
inv_classes = dict((cl, i) for i, cl in enumerate(classes))

def train(epoch, net, criterion, optimizer, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(testloader, criterion):
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

    return 100.*correct/total


def evaluate_f():
    '''evaluate the loss on the whole training dataset: no training. '''
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader_big):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item()
    return train_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1,
                        type=float, help='learning rate')
    parser.add_argument('--batchsize', default=128,
                        type=int, help='batch size')
    parser.add_argument('--model', default='VGG19', type=str, help='model name')
    parser.add_argument('--lr_mode', default='constant', type=str, help='lr mode')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    print('@@lr=', args.lr)
    print('@@batchsize=', args.batchsize)
    args.lr_mode=args.lr_mode.lower()
    print('lr mode=', args.lr_mode)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
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

    if args.model=='vgg19':
        net = VGG('VGG19')
    elif args.model=='resnet18':
        net = ResNet18()
    else:
        print('not run yet')
        sys.exit(1)
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(
            'checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.lr_mode == 'constant' or args.lr_mode == 'fixed':
        scheduler = None
    elif args.lr_mode=='schedule' or args.lr_mode=='anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        print('lr mode not supported this yet. ')

    # f_loss = []
    # torch.save(net.state_dict(), 'results/model0_vgg_sgd_alpha_'+str(args.lr)+'.pyc')#initial model
    acc_test = -1.0
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch, net, criterion, optimizer, trainloader)
        acc_test = test(testloader, criterion)
        # f_e = evaluate_f()
        # f_loss.append(f_e)
        if args.lr_mode=='schedule' or args.lr_mode=='anneal':
            scheduler.step()

    # f_loss = np.array(f_loss)
    # file_name='results/f_vgg_sgd_alpha_'+str(args.lr)+'.npy'
    # file_name = 'results/f_vgg_sgd_alpha_'+str(args.lr)+'_batchsize1024.npy'
    # file_name = 'results/f_resnet18_sgd_annealing_alpha'+str(args.lr)+'.npy'
    # np.save(file_name, f_loss)

    # torch.save(net.state_dict(), 'results/model_vgg_sgd_alpha_'+str(args.lr)+'.pyc')
    # torch.save(net.state_dict(), 'results/model_vgg_sgd_alpha_'
    #            + str(args.lr)+'_batchsize1024.pyc')
    print('final test acc:', acc_test)
    torch.save(net.state_dict(), 'results/model_' + args.model+ '_alpha_'+str(args.lr) +
               '_lrmode_'+ args.lr_mode +'_momentum_decayed_testacc_' + "{:.2f}".format(acc_test, 2)  +'.pyc')

    # torch.save(net.state_dict(), 'results/model_resnet18_annealing_alpha_'+str(args.lr)+'.pyc')

