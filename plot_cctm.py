import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
Just realizes these distance measures like CCD are variances
That's why they are good for charaterizing generalization. 

steps:
1. run measure_cross_class_distances.py to generate the cmp file
2. run this code to plot the cctm heat map
'''


def get_centers_and_cmp(model_name):
    # centers = np.load('./cctm/class_means_' + model_name + '_bn1.npy')#200, 10, 64: 200 epochs; 10 classes; 64 features
    cmp_final = np.load('./cctm/{}_class_cmp.npy'.format(model_name))/1000.# #to [0,1]; 200, 10, 10
    print('cmp_final.shape=', cmp_final.shape)
    return cmp_final

cmp_epoch_vgg = get_centers_and_cmp('vgg19')
plt.figure(1)
# plt.subplot(2, 2, 1)
ax = plt.gca()
# print('vgg recalls:')
# print(np.diag(np.diag(cmp_epoch_vgg[:, :])))
print('vgg CCTM:')
print(cmp_epoch_vgg[:, :])
#get rid of diagonals: the make the other entries very dim
cmp_vgg = cmp_epoch_vgg[:, :] - np.diag(np.diag(cmp_epoch_vgg[:, :]))
im = plt.imshow(cmp_vgg, cmap='hot_r')
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)
plt.title('CCTM: VGG19')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

plt.figure()
cmp_epoch_resnet = get_centers_and_cmp('resnet18')
# plt.subplot(2, 2, 2)
ax = plt.gca()
# print('res net recalls:')
# print(np.diag(np.diag(cmp_epoch_resnet[:, :])))
print('resnet CCTM:')
print(cmp_epoch_resnet[:, :])
cmp_resnet = cmp_epoch_resnet[:, :] - np.diag(np.diag(cmp_epoch_resnet[:, :]))
im = plt.imshow(cmp_resnet, cmap='hot_r')
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)
plt.title('CCTM: ResNet18')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)


plt.figure()
cmp_epoch_googlenet = get_centers_and_cmp('googlenet')
# plt.subplot(2, 2, 3)
ax = plt.gca()
# print('google net recalls:')
# print(np.diag(np.diag(cmp_epoch_googlenet[:, :])))
print('googlenet CCTM:')
print(cmp_epoch_googlenet)
cmp_googlenet = cmp_epoch_googlenet[:, :] - np.diag(np.diag(cmp_epoch_googlenet[:, :]))
im = plt.imshow(cmp_googlenet, cmap='hot_r')
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)
plt.title('CCTM: GoogleNet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

plt.figure()
cmp_epoch_dla = get_centers_and_cmp('simple_dla')
# plt.subplot(2, 2, 4)
ax = plt.gca()
# print('dla net recalls:')
# print(np.diag(np.diag(cmp_epoch_dla[:, :])))
print('dla cctm')
print(cmp_epoch_dla)
cmp_dla = cmp_epoch_dla[:, :] - np.diag(np.diag(cmp_epoch_dla[:, :]))
im = plt.imshow(cmp_dla, cmap='hot_r')
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)
plt.title('CCTM: DLA')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

plt.show()

#t-SNE?



