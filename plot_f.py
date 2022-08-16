import matplotlib.pyplot as pt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

f_loss01 = np.load('results/f_vgg_sgd_alpha_'+str(0.1)+'.npy')
f_loss001 = np.load('results/f_vgg_sgd_alpha_'+str(0.01)+'.npy')
f_loss0001 = np.load('results/f_vgg_sgd_alpha_'+str(0.001)+'.npy')
fc_loss0001 = np.load('results/fc_vgg_sgd_alpha_'+str(0.001)+'.npy')
print('fc_loss0001.shape=', fc_loss0001.shape)

rho01, rho001, rho0001, rho0001c = [], [], [], []
assert fc_loss0001.shape[1] == len(classes)
for i in range(fc_loss0001.shape[1]):
    rho0001c.append([])

# for i in range(len(f_loss01)):
#     rho01.append(f_loss01[i]/f_loss01[0])
#
# for i in range(len(f_loss001)):
#     rho001.append(f_loss001[i]/f_loss001[0])

for i in range(len(f_loss0001)):
    rho0001.append(f_loss0001[i]/f_loss0001[0])

for i in range(fc_loss0001.shape[0]):
    for cl in range(fc_loss0001.shape[1]):
        rho0001c[cl].append(fc_loss0001[i][cl]/fc_loss0001[0][cl])

# pt.plot(rho01, '-b',label='lr=0.1')
# pt.plot(rho001, '--r', label='lr=0.01')
# pt.plot(rho0001, '-r', label='lr=0.001')
pt.plot(rho0001, '-r', label=r'$\rho_t$')
pt.plot(rho0001c[-1], '-b', label=r'$\rho_t$--'+classes[-1])
pt.plot(rho0001c[1], '-g', label=r'$\rho_t$--'+classes[1])
pt.plot(rho0001c[3], '-c', label=r'$\rho_t$--'+classes[3])
pt.plot(rho0001c[5], '-m', label=r'$\rho_t$--'+classes[5])
pt.legend(fontsize=20)
pt.xlabel(r'$t$'+'(epochs)', fontsize=20)
pt.ylabel(r'$\rho_t$', fontsize=20)
pt.show()
