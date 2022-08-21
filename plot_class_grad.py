import matplotlib.pyplot as pt
import numpy as np

from main import classes

#a gradient is large: it is the almost same size as the networks
# model_path = 'results/model_vgg_sgd_alpha_' + str(0.01)

file = '/Users/hengshuaiyao/Downloads/cifar_10results/model_vgg_sgd_alpha_0.01_grad_norm2_classes.npy'

#cat gradient norm2 is not too large comparing to other classes.
grad_cls_norm2 = np.load(file)
pt.plot(grad_cls_norm2, '-b+')
pt.xticks(range(len(classes)), classes)
pt.show()


# for cl in range(len(classes)):
#     grad = np.load(model_path+'_grad_'+classes[cl]+'.npy')

