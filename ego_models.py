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
from main import classes,inv_classes, device
from utils import progress_bar

from model_gradient_classwise import find_model_file

def train_accuracy(net, data_loader):
    '''data_loader is the training data without transform
    this is to evaluate actually the accuracy on the training dataset
    thus we need to use net.eval() -- watch for batch normalization surprise.
    '''
    net.eval()
    train_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return 100. * correct / total #train_loss #report accuracy for a fixed scale


def train_accuracy_by_class(net, criterion, data_loader):
    net.eval()
    correct, total = 0, 0
    train_loss = 0
    class_correct, class_total = np.zeros(len(classes)), np.zeros(len(classes))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for cl in range(len(classes)):
                index = (targets == cl).nonzero()[:, 0]
                class_total[cl] += index.size(0)
                class_correct[cl] += predicted[index].eq(targets[index]).sum().item()

            progress_bar(batch_idx, len(data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    train_accuracy_class = np.zeros(len(classes))
    for cl in range(len(classes)):
        train_accuracy_class[cl] = 100. * class_correct[cl] / class_total[cl]
        print('@@class ', cl, ', accuracy:', train_accuracy_class[cl])
    return train_accuracy_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--model', default='VGG19', type=str, help='model name')
    parser.add_argument('--lr_mode', default='constant', type=str, help='lr mode')
    parser.add_argument('--batchsize', default=4096*2, type=int, help='batch size')
    parser.add_argument('--loss', default='gross', type=str, help='loss on the whole data or class-wise')

    parser.add_argument('--c1', default='cat', type=str, help='class name')
    parser.add_argument('--c2', default='dog', type=str, help='class name')
    parser.add_argument('--limit_theta', default=0.1, type=float, help='limit of theta1/theta')

    parser.add_argument('--resolution', default='low', type=str, help='resolution of the loss contour')


    args = parser.parse_args()
    args.model = args.model.lower()
    args.lr_mode = args.lr_mode.lower()

    if args.lr_mode == 'schedule':
        args.lr_mode = 'anneal'

    print('@@model=', args.model)
    print('@@lr=', args.lr)
    print('@@lr_mode=', args.lr_mode)
    print('@@limit_theta=', args.limit_theta)
    print('@@loss=', args.loss)
    # print('@@batchsize=', args.batchsize)

    if args.resolution == 'high':
        sigma_points = 10
    else:
        sigma_points = 5
    print('number of sigma points in each dimension:', sigma_points)

    if args.model == 'vgg19':
        net = VGG('VGG19')
    elif args.model == 'resnet18':
        net = ResNet18()
    else:
        print('not run yet')
        sys.exit(1)

    w_star = torch.nn.DataParallel(net)

    model_path = find_model_file('results/', args.model, args.lr, args.lr_mode)

    print('loading model at path:', model_path)
    w_star.load_state_dict(torch.load(model_path))
    # print(w_star)

    print('loading class gradients for classes:')
    c1 = inv_classes[args.c1.lower()]
    c2 = inv_classes[args.c2.lower()]
    print('c1=', c1, '; c2=', c2)
    c1_grad = pickle.load(open(model_path + '_grad_' + classes[c1] + '.pkl', "rb"))
    c2_grad = pickle.load(open(model_path + '_grad_' + classes[c2] + '.pkl', "rb"))

    optimizer = optim.SGD(w_star.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()  # by default. it's mean.



    #oh. I found constant lr=0.01 leads to a much sharp minima in the interference space; while lr=0.1 with scheduling it is much more flat

    theta1s = np.linspace(0, args.limit_theta, sigma_points)

    theta1s_neg = -theta1s[1:]
    theta1s = list(reversed(theta1s_neg.tolist())) + theta1s.tolist()
    print('theta1s=', theta1s)

    #Data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_test)  # use transform_test
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=False, num_workers=2)

    acc_star = train_accuracy(w_star, trainloader)
    print('acc_star=', acc_star)

    theta2s = theta1s

    if args.loss == 'gross':
        acc = -np.ones((len(theta1s), len(theta2s)))
    else:
        acc = -np.ones((len(classes), len(theta1s), len(theta2s)))

    with torch.no_grad():
        for i, theta1 in enumerate(theta1s):
            print('####i=', i)
            for j, theta2 in enumerate(theta2s):
                print('----j=', j)
                w = copy.deepcopy(w_star)
                for name, param in w.named_parameters():
                    param.add_(c1_grad[name], alpha=-theta1)
                    param.add_(c2_grad[name], alpha=-theta2)
                #todo check if the w changes

                #evaluate w
                if args.loss == 'gross':
                    acc[i, j] = train_accuracy(w, trainloader)
                    # print('accuracy=', acc[i, j])
                else:
                    acc[:, i,j] = train_accuracy_by_class(w, trainloader)

    if args.loss == 'gross':
        print('accuracy=', acc)
        np.save(model_path +'_' + args.c1 + '_' + args.c2 +'_egomodels_acc_limit_theta' + str(args.limit_theta) + '.npy', np.array(acc))
    else:
        np.save(model_path + '_' + args.c1 + '_' + args.c2 + '_egomodels_acc_limit_theta' + str(args.limit_theta) + '_classwise.npy', np.array(acc))
        print('accuracy of CAT=', acc[3, :, :])












