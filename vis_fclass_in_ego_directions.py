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
parser.add_argument('--limit_theta', default=0.1, type=float, help='limit of theta1/theta2')
parser.add_argument('--c1', default='cat', type=str, help='dimension 1 of the loss visualization space')
parser.add_argument('--c2', default='dog', type=str, help='dimension 2 of the loss visualization space')
parser.add_argument('--c', default='cat', type=str, help='whose class loss do you want to examine')

parser.add_argument('--resolution', default='low', type=str, help='resolution of the loss contour')

args = parser.parse_args()
args.model = args.model.lower()
args.lr_mode = args.lr_mode.lower()

limit_theta = 1.

'''
This visualization shows that CAR causes all class losses increase sharply. 
Interesting. Need to understand better. 

CAT and dog ego directions don't influence CAR, TRUCK and DEER's loss: 
So CAR interferes with CAT most?
My concern: the conclusion is different based on the choice of Theta_1 and Theta_2
So better to use some sort of objective measure: like on the training data. 
'''

print('visualzing loss for ego directions of classes:')
# c1 = inv_classes[args.c1.lower()]
# c2 = inv_classes[args.c2.lower()]
# print('c1=', c1, '; c2=', c2)
c = inv_classes[args.c.lower()]
print('c=', c)

#CAT loss in CAT-DOG spaces
# file ='results/model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_cat_dog_egomodels_acc_limit_theta1.0_classwise.npy'
# c1, c2 = 3, 5

#CAT loss in CAR-TRUCK spaces
file ='results/model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_car_truck_egomodels_acc_limit_theta1.0_classwise.npy'
c1, c2 = 1, 9

print('loading file ', file)
acc = np.load(file)

# c = 3
# c = 5
# c = 1
# c = 9
#c = 2
# c = 3

print('visualizing classs {} loss in {}-{} spaces'.format(classes[c].upper(), classes[c1].upper(), classes[c2].upper()))

print(acc.shape)

#only show positive theta
acc = acc[:, 4:, 4:]
print(acc.shape)
#X and Y plane: The loss was two-loop. outside is theta1 fixed. theta2 changes. So theta1 is y. c1 is y.
# theta1s = np.linspace(0, limit_theta, 10)
theta1s = np.linspace(0, limit_theta, 5)
theta1s_neg = -theta1s[1:]
# theta1s = list(reversed(theta1s_neg.tolist())) + theta1s.tolist()
theta1s = theta1s.tolist() #cut negatives
print('theta1s=', theta1s)
theta2s = theta1s

losses = 100 - acc

plt.figure(1)
print(losses[c])
print(losses[c].shape)
plt.imshow(losses[c], interpolation='none')
# plt.xticks([-limit_theta, -limit_theta/2., 0, limit_theta/2., limit_theta])
# plt.yticks([-limit_theta, -limit_theta/2., 0, limit_theta/2., limit_theta])
plt.xlabel(classes[c2].upper() + ' ego direction')
plt.ylabel(classes[c1].upper() + ' ego direction')

hf = plt.figure(2)
ha = hf.add_subplot(111, projection='3d')
X, Y = np.meshgrid(theta1s, theta2s)
ha.plot_surface(X, Y, losses[c], cmap=cm.jet,  edgecolor='darkred', linewidth=0.1, rstride=1, cstride=1)
plt.xlabel(classes[c2].upper() + ' ego direction')
plt.ylabel(classes[c1].upper() + ' ego direction')
ha.set_zlabel('100 - training accuracy of '+ classes[c].upper())
ha.set_zlim([0,100])

plt.show()
