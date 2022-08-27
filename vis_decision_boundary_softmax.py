import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
import argparse
import pickle
from main import classes, inv_classes

header_path = 'results/model_vgg19_alpha_0.1_lrmode_schedule_momentum_decayed_testacc_93.87.pyc_softmax_'

softmax = -np.ones((len(classes), 5000, len(classes)))
for cl in range(len(classes)):
    class_name = classes[cl]
    file = header_path + class_name +'.pkl'
    # print('loading soft max file:')
    print(file)
    softmax_cl = pickle.load(open(file, 'rb'))
    print(softmax_cl.shape)
    print(np.min(softmax_cl))
    print(np.max(softmax_cl))
    assert 0 < np.min(softmax_cl) and np.min(softmax_cl) < 1.
    assert 0 < np.max(softmax_cl) and np.max(softmax_cl) < 1.
    softmax[cl, :, :] = softmax_cl

print(softmax.shape)
print(np.min(softmax))
print(np.max(softmax))

#it appears that there are a few samples of cats whose CAT prediction is very low.
# hi = np.histogram(softmax[3, :, 3], bins=10)
# print(hi)

plt.figure(1)
#probabilities that cats are predicted as CAT
plt.hist(softmax[3, :, 3], bins=list(np.linspace(0.998, 1.0, 10)))

plt.figure(2)
#probabilities that cats are predicted as DOG
plt.hist(softmax[3, :, 5], bins=list(np.linspace(0., 0.001, 10)))
plt.show()

'''
Conclusion: there are no ties for the cats in predicting them as CAT or DOG 
So the decision boundary is empty according to ties in label prediction. 
'''