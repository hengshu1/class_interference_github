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

from model_gradient_classwise import find_model_file, train_loader_class, compute_sample_softmax


'''
study the decision boundary of a trained model by gradient ascent attacks

'''

def train_ascent(net, criterion, optimizer, class_loader):
    '''class_loader is the training data for a class without transform
    Compute the softmax for each sample for the class objects
    '''
    net.train()
    train_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(class_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = - criterion(outputs, targets) #ascent attack

            #update the model
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(class_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--model', default='VGG19', type=str, help='model name')
    parser.add_argument('--batchsize', default=4096*2, type=int, help='batch size')

    parser.add_argument('--limit_theta', default=0.1, type=float, help='limit of theta1/theta')

    parser.add_argument('--resolution', default='low', type=str, help='resolution of the loss contour')


    args = parser.parse_args()
    args.model = args.model.lower()

    print('@@model=', args.model)
    print('@@lr=', args.lr)
    print('@@lr_mode=', args.lr_mode)
    print('@@limit_theta=', args.limit_theta)
    # print('@@batchsize=', args.batchsize)

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

    optimizer = optim.SGD(w_star.parameters(), lr=args.lr, momentum=0., weight_decay=0)
    criterion = nn.CrossEntropyLoss()  # by default. it's mean.

    for cl in range(len(classes)):
        print('attack class: ', classes[cl])
        trainloader_cls, num_samples_cls = train_loader_class(label=cl)
        print('loading class samples:', num_samples_cls)

        net = copy.deepcopy(w_star)

        #train a small number of steps
        for epoch in range(5):
            train_ascent(net, criterion, optimizer, trainloader_cls)

        softmax_cls = compute_sample_softmax(net, trainloader_cls, num_samples_cls)
        f = open(model_path + '_softmax_attack_class' + classes[cl] + '_by_ascent.pkl', "wb")
        pickle.dump(softmax_cls, f)
        f.close()

    # if args.loss == 'gross':
    #     print('accuracy=', acc)
    #     np.save(model_path +'_' + args.c1 + '_' + args.c2 +'_egomodels_acc_limit_theta' + str(args.limit_theta) + '.npy', np.array(acc))
    # else:
    #     np.save(model_path + '_' + args.c1 + '_' + args.c2 + '_egomodels_acc_limit_theta' + str(args.limit_theta) + '_classwise.npy', np.array(acc))
    #     print('accuracy of CAT=', acc[3, :, :])












