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

def train_ascent(net, criterion, optimizer, class_loader):
    '''class_loader is the training data for a class without transform
    Compute the softmax for each sample for the class objects
    the BN has a secret surprise here. If we train using samples for each class.
    the mean trained using the whole training data is not going to generalize to class samples.
    So the even the step() is turned off, the accuracy drops off in a great deal.
    '''
    # net.train()
    net.eval()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(class_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = - criterion(outputs, targets) #ascent attack
        # loss = criterion(outputs, targets)

        #update the model
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(class_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def train_ascent_per_sample(net, criterion, optimizer, class_loader):
    '''
    debugging: just one sample: how many steps to cross the boundary?
    how many epochs does it take for each sample's label prediction to switch? And do cats switch to DOG?
    class_loader: batchsize = 1
    found: easily diverges. 
    '''
    # net.train()
    net.eval()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(class_loader):
        print('batch_idx=', batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)

        # inputs = inputs[0].unsqueeze(0)
        # targets = targets[0].unsqueeze(0)
        # print('inputs.shape=', inputs.shape)
        # print('targets.shape=', targets.shape)

        for epoch in range(10):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = - criterion(outputs, targets) #ascent attack
            # loss = criterion(outputs, targets)

            #update the model
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)

            print('class:', targets[0])
            print('predi:', predicted[0])
            print('loss:', train_loss)

        # print('predicted=', predicted)
        # break


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

    # if args.model == 'vgg19':
    #     w_star = VGG('VGG19')
    # elif args.model == 'resnet18':
    #     w_star = ResNet18()
    # else:
    #     print('not run yet')
    #     sys.exit(1)

    # w_star = w_star.to(device)
    # if device == 'cuda':
    #     w_star = torch.nn.DataParallel(w_star)
    #     cudnn.benchmark = True
    # w_star = w_star.to(device)

    model_path = find_model_file('results/', args.model, args.lr, args.lr_mode)

    print('loading model at path:', model_path)
    # w_star.load_state_dict(torch.load(model_path))
    # print(w_star)


    for cl in range(len(classes)):
        print('attack class: ', classes[cl])
        # trainloader_cls, num_samples_cls = train_loader_class(label=cl, batch_size=128)
        trainloader_cls, num_samples_cls = train_loader_class(label=cl, batch_size=1)
        # print('loading class samples:', num_samples_cls)

        net = VGG('VGG19')
        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        net.load_state_dict(torch.load(model_path))

        # for name, param in net.named_parameters():
        #     print('name=', name)
        #     print('requires_grad=', param.requires_grad)

        # optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
        optimizer = optim.SGD(net.parameters(), lr=.0001, momentum=0., weight_decay=0.)
        criterion = nn.CrossEntropyLoss()  # by default. it's mean.
        # print('testing the original trained model')
        # test(net, trainloader_cls, criterion)
        # test(net, trainloader, criterion)
        # print('done testing.')
        #train a small number of steps

        # train_ascent(net, criterion, optimizer, trainloader_cls)
        train_ascent_per_sample(net, criterion, optimizer, trainloader_cls)
        # attack all the samples
        # train_ascent(net, criterion, optimizer, trainloader)

        softmax_cls = compute_sample_softmax(net, criterion, trainloader_cls, num_samples_cls)
        f = open(model_path + '_softmax_attack_class' + classes[cl] + '_by_ascent.pkl', "wb")
        pickle.dump(softmax_cls, f)
        f.close()

    # if args.loss == 'gross':
    #     print('accuracy=', acc)
    #     np.save(model_path +'_' + args.c1 + '_' + args.c2 +'_egomodels_acc_limit_theta' + str(args.limit_theta) + '.npy', np.array(acc))
    # else:
    #     np.save(model_path + '_' + args.c1 + '_' + args.c2 + '_egomodels_acc_limit_theta' + str(args.limit_theta) + '_classwise.npy', np.array(acc))
    #     print('accuracy of CAT=', acc[3, :, :])












