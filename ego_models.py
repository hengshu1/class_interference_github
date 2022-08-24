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
from main import classes, device
from utils import progress_bar

def train_loss(net, data_loader):
    '''data_loader is the training data without transform'''
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    #constant lr experiment
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')

    #lr scheduling experiment used alpha_0 = 0.1 for initial lr
    # parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

    parser.add_argument('--batchsize', default=4096*2, type=int, help='batch size')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    print('@@lr=', args.lr)
    print('@@batchsize=', args.batchsize)

    w_star = VGG('VGG19')
    # w_star = ResNet18()

    w_star = torch.nn.DataParallel(w_star)

    # model_path = 'results/model_vgg_sgd_alpha_'+str(args.lr)
    # model_path = 'results/model_vgg_sgd_alpha_'+str(0.001)+'_batchsize1024'
    model_path = 'results/model_vgg19_alpha_' + str(args.lr) + '_momentum_decayed'
    # model_path = 'results/model_resnet18_annealing_alpha_' + str(args.lr)
    # model_path = 'results/model_resnet18_alpha_' + str(args.lr) + '_momentum_decayed'


    print('loading model at path:', model_path)
    w_star.load_state_dict(torch.load(model_path+'.pyc'))
    # print(w_star)

    # optimizer = optim.SGD(w_star.parameters(), lr=args.lr, momentum=0, weight_decay=0)  # vannila SGD
    optimizer = optim.SGD(w_star.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()  # by default. it's mean.

    limit_theta = 0.1

    #oh. I found constant lr=0.01 leads to a much sharp minima in the interference space; while lr=0.1 with scheduling it is much more flat
    theta1s = np.linspace(0, limit_theta, 10)#high resolution
    # theta1s = np.linspace(0, limit_theta, 5)#low resolutions

    theta1s_neg = -theta1s[1:]
    theta1s = list(reversed(theta1s_neg.tolist())) + theta1s.tolist()
    print('theta1s=', theta1s)

    # c1, c2 = 3, 5#CAT DOG
    c1, c2 = 1, 9 #CAR TRUCK
    # c1, c2 = 7, 8  # Horse Ship

    c1_grad = pickle.load(open(model_path + '_grad_' + classes[c1] + '.pkl', "rb"))
    c2_grad = pickle.load(open(model_path + '_grad_' + classes[c2] + '.pkl', "rb"))

    # print('type(cat_grad)=', type(cat_grad))
    # print('cat_grad=', cat_grad)
    # sys.exit(1)

    #Data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_test)  # use transform_test
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=False, num_workers=2)

    loss_star = train_loss(w_star, trainloader)
    print('loss_star=', loss_star)

    theta2s = theta1s
    losses = -np.ones((len(theta1s), len(theta2s)))

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
                losses[i, j] = train_loss(w, trainloader)
                print('loss=', losses[i, j])

    np.save(model_path+'_' + classes[c1] + '_'+classes[c2]+'_egomodels_acc_limit_theta' + str(limit_theta) + '.npy', np.array(losses))
    print('losses=', losses)











