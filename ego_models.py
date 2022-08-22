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

    return train_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01,
                        type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    print('@@lr=', args.lr)

    w_star = VGG('VGG19')

    model_path = 'results/model_vgg_sgd_alpha_'+str(args.lr)
    # model_path = 'results/model_vgg_sgd_alpha_'+str(0.001)+'_batchsize1024'
    print('loading model at path:', model_path)
    w_star.load_state_dict(torch.load(model_path+'.pyc'))
    print(w_star)

    criterion = nn.CrossEntropyLoss()  # by default. it's mean.
    optimizer = optim.SGD(w_star.parameters(), lr=args.lr, momentum=0, weight_decay=0)  # vannila SGD

    cat_grad = pickle.load(open(model_path + '_grad_' + classes[3] + '.pkl', "rb"))
    dog_grad = pickle.load(open(model_path + '_grad_' + classes[5] + '.pkl', "rb"))

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

    alpha = args.lr
    theta1s = [0, 1e-3, 1e-1]#[1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0]#xrange
    theta2s = theta1s
    losses = -np.ones(len(theta1s), len(theta2s))
    for i, theta1 in enumerate(theta1s):
        for j, theta2 in enumerate(theta2s):
            w = copy.deepcopy(w_star)
            for name, param in w.named_parameters():
                param.add_(cat_grad[name], alpha=-theta1)
                param.add_(dog_grad[name], alpha=-theta2)
            #todo check if the w changes

            #evaluate w
            losses[i, j] = train_loss(w, trainloader)

    print('losses=', losses)











