import torch
import torch.nn as nn
from models import *
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import argparse
import time
import sys
import pickle
import torch.backends.cudnn as cudnn
from main import classes,inv_classes, device, test
from utils import progress_bar

from model_gradient_classwise import find_model_file, train_loader_class, compute_sample_softmax


'''
study the decision boundary of a trained model by gradient ascent attacks

'''

def label_attack(net, criterion, optimizer, train_loader, c1=3, c2=5):
    '''
    use all the training data in the train_loader
    Attack the label: change objects of c1 to c2
    default is to study c1=CAT and c2 = DOG
    '''

    print('attack {} objects with labeling {}'.format(classes[c1].upper(), classes[c2].upper()))

    net.train()
    # net.eval()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        index = (targets == c1).nonzero()[:, 0]
        targets[index] = torch.ones_like(index) * c2

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        #update the model
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--model', default='VGG19', type=str, help='model name')
    parser.add_argument('--lr_mode', default='schedule', type=str, help='lr mode')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size')
    parser.add_argument('--loss', default='gross', type=str, help='loss on the whole data or class-wise')
    parser.add_argument('--resolution', default='low', type=str, help='resolution of the loss contour')

    args = parser.parse_args()
    args.model = args.model.lower()

    print('@@model=', args.model)
    print('@@lr=', args.lr)
    print('@@lr_mode=', args.lr_mode)
    # print('@@batchsize=', args.batchsize)

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

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    model_path = find_model_file('results/', args.model, args.lr, args.lr_mode)

    print('loading model at path:', model_path)
    net = VGG('VGG19')
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.load_state_dict(torch.load(model_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0, weight_decay=0)

    #it appears that cats and dogs are not easy to be attacked. they are resillient to attack.
    #however, the most resilient is car. only 9 cats switch their sign.
    #so you need more samples on the boundary?
    #quite unsymmetric
    #ships and horses are much easier to give up; especially ships
    # c1, c2 = 3, 5
    # c1, c2 = 5, 3
    # c1, c2 = 1, 9
    # c1, c2 = 9, 1
    # c1, c2 = 7, 8
    c1, c2 = 8, 7
    object_cls_c1, num_samples_cls_c1 = train_loader_class(label=c1, batch_size=5000)
    object_cls_c2, num_samples_cls_c2 = train_loader_class(label=c2, batch_size=5000)
    print('num_samples_cls_c1=',num_samples_cls_c1)
    print('num_samples_cls_c2=', num_samples_cls_c2)

    for epoch in range(1):
        label_attack(net, criterion, optimizer, trainloader, c1=c1, c2=c2)
        print('testing c1={}'.format(classes[c1]))
        acc_test_c1 = test(net, object_cls_c1, criterion)
        print('testing for all classes')
        for c_test in range(len(classes)):
            print('testing {}'.format(classes[c_test]))
            object_cls_ctest, num_samples_cls_ctest = train_loader_class(label=c_test, batch_size=5000)
            acc_test_ctest = test(net, object_cls_ctest, criterion)

    # softmax_cls = compute_sample_softmax(net, criterion, trainloader_cls, num_samples_cls)
    # f = open(model_path + '_softmax_attack_class' + classes[cl] + '_by_ascent.pkl', "wb")
    # pickle.dump(softmax_cls, f)
    # f.close()

    # if args.loss == 'gross':
    #     print('accuracy=', acc)
    #     np.save(model_path +'_' + args.c1 + '_' + args.c2 +'_egomodels_acc_limit_theta' + str(args.limit_theta) + '.npy', np.array(acc))
    # else:
    #     np.save(model_path + '_' + args.c1 + '_' + args.c2 + '_egomodels_acc_limit_theta' + str(args.limit_theta) + '_classwise.npy', np.array(acc))
    #     print('accuracy of CAT=', acc[3, :, :])












