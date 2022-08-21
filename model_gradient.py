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
# from .measure_cross_class_distances import get_train_cats
from main import device

torch.manual_seed(0)

def concat_param_grad(net):
    '''
    fix x = net; we are interested at the final model centered. compute f'(x)
    And check the curvatures around it; and see any direction going down? any leaky direction? If mu<0, then it's leaky
    todo: follow https://discuss.pytorch.org/t/get-the-gradient-of-the-network-parameters/50575/2
    '''
    wg_all = None
    for name, param in net.named_parameters():
        # print('name=', name)
        # print(name, param.size())
        # print(name, 'grad size:', param.grad.size())
        # print('param.grad.shape=', param.grad.shape)
        # if name == 'module.features.0.weight':
        #     print('concat_param_grad: param.grad=', param.grad.cpu().numpy()[:3, 0, 0, 0])
        _wgrad = to_vector(param.grad)
        if wg_all is None:
            wg_all = _wgrad
        else:
            wg_all = torch.cat((wg_all, _wgrad))

    # print('total params from w_all=', wg_all.size())
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    # print('total params =', pytorch_total_params)  # same
    return wg_all


def aver_grad_1D(trainloader, net, optimizer, criterion):
    '''average the gradient into 1D vector'''
    # net.train()
    train_loss = 0
    correct = 0
    total = 0

    grads_sum = None
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # print('targets=', targets)
        loss = criterion(outputs, targets)
        loss.backward()#compute gradient
        # optimizer.step()#no model update

        if batch_idx == 0:
            print('targets=', targets)
            for name, param in net.named_parameters():
                print('name=', name)
                print(param.grad[:3, 0, 0, 0])
                sys.exit(1)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        grads = concat_param_grad(net)
        if grads_sum is None:
            grads_sum = grads
        else:
            grads_sum += grads

    grads_sum /= total
    print('total samples=', total)
    return grads_sum



def aver_grad_net(trainloader, net, optimizer, criterion):
    '''average the gradient and keeps it in a model form'''
    # net.train()
    train_loss = 0
    correct = 0
    total = 0

    grads_sum = {}
    for name, param in net.named_parameters():
        grads_sum[name] = None
    # print('grads_sum=', grads_sum)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # optimizer.step()

        if batch_idx == 0:
            print('targets=', targets)
            for name, param in net.named_parameters():
                print('name=', name)
                print(param.grad[:3, 0, 0, 0])
                break

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        for name, param in net.named_parameters():
            # print('name=', name)
            # print('param.grad=', param.grad.cpu().numpy()[:3, 0, 0, 0])
            if grads_sum[name] is None:
                grads_sum[name] = param.grad
            else:
                grads_sum[name] += param.grad

    print('total samples=', total)
    for name in grads_sum.keys():
        grads_sum[name] /= total

    return grads_sum


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='learning rate')
    parser.add_argument('--batchsize', default=128,
                        type=int, help='batch size')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    print('@@lr=', args.lr)
    print('@@batchsize=', args.batchsize)

    net = VGG('VGG19')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = net.to(device)
    # if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        # cudnn.benchmark = True

    net1 = copy.deepcopy(net)

    # model_path = 'results/model_vgg_sgd_alpha_'+str(0.001)
    model_path = 'results/model_vgg_sgd_alpha_'+str(0.01)
    # model_path = 'results/model_vgg_sgd_alpha_'+str(0.001)+'_batchsize1024'
    # model_path = 'results/model_vgg_sgd_alpha_'+str(0.01)+'_batchsize1024'
    # print('loading model at path:', model_path)
    # net.load_state_dict(torch.load(model_path+'.pyc'))
    # print(net)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    criterion = nn.CrossEntropyLoss(reduction='sum')  # by default. it's mean.
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0, weight_decay=0)  # first do without momentum

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=False, num_workers=2)

    print('aver_grad_net')
    grad_net = aver_grad_net(trainloader, net, optimizer, criterion)

    #reset the model
    net2 = net1
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0, weight_decay=0)
    criterion2 = nn.CrossEntropyLoss(reduction='sum')

    print('@@@@')
    print('aver_grad_1D')
    grad_1D = aver_grad_1D(trainloader, net2, optimizer2, criterion2)
    print('grad_1D.shape=', grad_1D.shape)

    #why the two gradients are different????
    # grads = []
    # for name in grad_net.keys():
    #     grads.append(grad_net[name].view(-1))
    #     # grads.append(to_vector(grad_net[name]))
    # grads = torch.cat(grads)

    # print('grad diff=', torch.norm(grads - grad_1D))

    # np.save(model_path+'_grad.npy', grad)

    # class_loader = get_train_cats(trainset, batch_size=1024, label=3)  # get cats
    #how many cats?