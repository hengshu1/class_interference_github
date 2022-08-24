import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from main import classes
import sys

# c1, c2 = 3, 5#CAT DOG
c1, c2 = 1, 9  # CAR TRUCK
# c1, c2 = 7, 8  # Horse Ship

limit_theta = 0.1

#VGG
# model = 'VGG19'

#Resnet18
model = 'resnet18'


model_path = 'results/model_' + model + '_alpha_' + str(0.01) + '_momentum_decayed'
print(model_path)
file = model_path+'_' + classes[c1] + '_'+classes[c2]+'_egomodels_acc_limit_theta' + str(limit_theta) + '.npy'

print('loading file ', file)

losses = np.load(file)
#for accuracy on the training error we used percentage of training error
losses =  100 - losses

#X and Y plane: The loss was two-loop. outside is theta1 fixed. theta2 changes. So theta1 is y. c1 is y.
theta1s = np.linspace(0, limit_theta, 10)
# theta1s = np.linspace(0, limit_theta, 5)
theta1s_neg = -theta1s[1:]
theta1s = list(reversed(theta1s_neg.tolist())) + theta1s.tolist()
print('theta1s=', theta1s)
theta2s = theta1s

plt.figure(1)
print(losses)
print(losses.shape)
plt.imshow(losses, interpolation='none')
plt.xticks([-0.1, -0.05, 0, 0.05, 0.1])
plt.yticks([-0.1, -0.05, 0, 0.05, 0.1])
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
ha.set_zlabel('100 - training accuracy')
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
