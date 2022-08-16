import matplotlib.pyplot as pt
import numpy as np


f_loss01 = np.load('results/f_vgg_sgd_alpha_'+str(0.1)+'.npy')
f_loss001 = np.load('results/f_vgg_sgd_alpha_'+str(0.01)+'.npy')
# f_loss0001 = np.load('results/f_vgg_sgd_alpha_'+str(0.001)+'.npy')
print('f_loss001.shape=',f_loss001.shape)

rho01,rho001,rho0001 = [], [], []
for i in range(len(f_loss01)):
    rho01.append(f_loss01[i]/f_loss01[0])

for i in range(len(f_loss001)):
    rho001.append(f_loss001[i]/f_loss001[0])

# for i in range(len(f_loss0001)):
#     rho0001.append(f_loss0001[i]/f_loss0001[0])

pt.plot(rho01, '-b',label='lr=0.1')
pt.plot(rho001, '--r', label='lr=0.01')
# pt.plot(rho0001, '-.k', label='lr=0.001')
pt.legend(fontsize=20)
pt.xlabel(r'$t$'+'(epochs)',fontsize=20)
pt.ylabel(r'$\rho_t$',fontsize=20)
pt.show()
