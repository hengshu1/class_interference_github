import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
import argparse
from main import classes, inv_classes
from model_gradient_classwise import find_model_file

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--model', default='VGG19', type=str, help='model name')
parser.add_argument('--lr_mode', default='constant', type=str, help='lr mode')
parser.add_argument('--batchsize', default=4096 * 2, type=int, help='batch size')
parser.add_argument('--limit_theta', default=0.01, type=float, help='limit of theta1/theta2')
parser.add_argument('--c1', default='cat', type=str, help='class name of ego direction 1')
parser.add_argument('--c2', default='dog', type=str, help='class name of ego direction 2')

parser.add_argument('--resolution', default='low', type=str, help='resolution of the loss contour')

args = parser.parse_args()
args.model = args.model.lower()
args.lr_mode = args.lr_mode.lower()

print('visualzing loss for ego directions of classes:')
c1 = inv_classes[args.c1.lower()]
c2 = inv_classes[args.c2.lower()]
print('c1=', c1, '; c2=', c2)

# model_path = find_model_file('results/', args.model, args.lr, args.lr_mode)

#todo: this needs to hand change; command line model selection does not work
# model_path = 'results/model_vgg19_alpha_0.0001_lrmode_constant_momentum_decayed_testacc_84.99.pyc'
# model_path = 'results/model_resnet18_alpha_0.0001_lrmode_constant_momentum_decayed_testacc_86.88.pyc'
# model_path = 'results/model_resnet18_alpha_0.1_lrmode_anneal_momentum_decayed_testacc_95.15.pyc'
# model_path = 'results/model_vgg19_alpha_0.01_lrmode_constant_momentum_decayed_testacc_88.76.pyc'
# model_path = 'results/model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc'
# file = 'results/model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_cat_dog_egomodels_acc_limit_theta1.0.npy'
# file = 'results/model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_cat_dog_egomodels_acc_limit_theta1.0_2nd.npy'
file = 'results/model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_car_truck_egomodels_acc_limit_theta1.0_2nd.npy'
# print('model_path', model_path)
# file = 'results/model_vgg19_alpha_0.0001_lrmode_constant_momentum_decayed_testacc_84.99.pyc_cat_dog_egomodels_acc_limit_theta0.01.npy'
# file = 'results/model_vgg19_alpha_0.0001_lrmode_constant_momentum_decayed_testacc_84.99.pyc_cat_dog_egomodels_acc_limit_theta0.01_2nd.npy'
# file ='results/model_vgg19_alpha_0.0001_lrmode_constant_momentum_decayed_testacc_84.99.pyc_car_truck_egomodels_acc_limit_theta0.01_2nd.npy'
# file ='results/model_vgg19_alpha_0.0001_lrmode_constant_momentum_decayed_testacc_84.99.pyc_horse_ship_egomodels_acc_limit_theta0.01_2nd.npy'
# file = 'results/model_vgg19_alpha_0.01_lrmode_constant_momentum_decayed_testacc_88.76.pyc_cat_dog_egomodels_acc_limit_theta0.1_2nd.npy'
limit_theta = args.limit_theta
# file = model_path +'_' + args.c1 + '_' + args.c2 +'_egomodels_acc_limit_theta' + str(limit_theta) + '.npy'

# file ='results/model_resnet18_alpha_0.1_lrmode_anneal_momentum_decayed_testacc_95.15.pyc_cat_dog_egomodels_acc_limit_theta1.0.npy'
# file = 'results/model_resnet18_alpha_0.1_lrmode_anneal_momentum_decayed_testacc_95.15.pyc_car_truck_egomodels_acc_limit_theta1.0.npy'
# file = 'results/model_resnet18_alpha_0.1_lrmode_anneal_momentum_decayed_testacc_95.15.pyc_horse_ship_egomodels_acc_limit_theta1.0.npy'
# file = 'results/model_resnet18_alpha_0.0001_lrmode_constant_momentum_decayed_testacc_86.88.pyc_cat_dog_egomodels_acc_limit_theta0.01_2nd.npy'

print('loading file ', file)
acc = np.load(file)
#for accuracy on the training error we used percentage of training error

losses = 100 - acc

#X and Y plane: The loss was two-loop. outside is theta1 fixed. theta2 changes. So theta1 is y. c1 is y.
# theta1s = np.linspace(0, limit_theta, 10)
theta1s = np.linspace(0, limit_theta, 5)
theta1s_neg = -theta1s[1:]
theta1s = list(reversed(theta1s_neg.tolist())) + theta1s.tolist()
print('theta1s=', theta1s)
theta2s = theta1s

plt.figure(1)
print(losses.shape)
plt.imshow(losses, interpolation='none')
plt.xticks([-limit_theta, -limit_theta/2., 0, limit_theta/2., limit_theta])
plt.yticks([-limit_theta, -limit_theta/2., 0, limit_theta/2., limit_theta])
plt.xlabel(classes[c2] + ' ego direction')
plt.ylabel(classes[c1] + ' ego direction')

hf = plt.figure(2)
ha = hf.add_subplot(111, projection='3d')
# x = range(losses.shape[0])
# y = range(losses.shape[1])
# X, Y = np.meshgrid(x, y)
X, Y = np.meshgrid(theta1s, theta2s)
# ha.plot_surface(X, Y, losses, cmap=cm.hsv)
ha.plot_surface(X, Y, losses, cmap=cm.jet,  edgecolor='darkred', linewidth=0.1, rstride=1, cstride=1)
plt.xlabel(classes[c2] + ' ego direction')
plt.ylabel(classes[c1] + ' ego direction')
ha.set_zlabel('100 - recall accuracy')
ha.set_zlim([0,100])
plt.show()

# def f(x, y):
#     return np.sin(np.sqrt(x ** 2 + y ** 2))
#
# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)
#
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)
#
# print(X.shape)
# print(Y.shape)
# print(Z.shape)
# print(X[0, :])
# print(X[1, :])#the same
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z');
