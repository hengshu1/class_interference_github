import torch
import torch.nn as nn
from models import *
import torch.optim as optim
import numpy as np
import copy
import argparse
import time
import sys
import torch.backends.cudnn as cudnn
from model_gradient_classwise import train_loader_class

from main import classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    print('@@lr=', args.lr)

    net = VGG('VGG19')

    # model_path = 'results/model_vgg_sgd_alpha_'+str(0.1)
    model_path = 'results/model_vgg_sgd_alpha_'+str(0.01)
    # model_path = 'results/model_vgg_sgd_alpha_'+str(0.001)
    # model_path = 'results/model_vgg_sgd_alpha_'+str(0.001)+'_batchsize1024'
    print('loading model at path:', model_path)
    net.load_state_dict(torch.load(model_path+'.pyc'))
    print(net)

    criterion = nn.CrossEntropyLoss(reduction='sum')  # by default. it's mean.
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0, weight_decay=0)  # vannila SGD

    #This computes the gradient for each class
    for cl in range(len(classes)):
        print('class: ', classes[cl])
        trainloader_cls = train_loader_class(label=cl)

        grad = aver_grad(trainloader_cls, net, optimizer, criterion)
        print('grads.shape=', grad.shape)
        np.save(model_path + '_grad_' + classes[cl] + '.npy', grad)



