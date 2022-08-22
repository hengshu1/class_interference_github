import numpy as np
import matplotlib.pylab as plt

file = model_path = 'results/model_vgg_sgd_alpha_'+str(0.01) + '_cat_dog_egomodels_loss.npy'
losses = np.load(file)

print(losses.shape)

