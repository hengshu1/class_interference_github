import matplotlib.pyplot as pt
import numpy as np

lr=0.1
file_name='results/f_vgg_sgd_alpha_'+str(lr)+'.npy'
f_loss = np.load(file_name)
print('f_loss.shape=', f_loss)#1D?

rho=[]
for i in range(len(f_loss)):
    rho.append(f_loss[i]/f_loss[0])
pt.plot(rho, '-b')
pt.show()
