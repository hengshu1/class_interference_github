import matplotlib.pyplot as pt
import numpy as np

file_name='results/f_sgd_alpha_'+str(args.lr)+'.npy'
f_loss = np.load(filename)
print('f_loss.shape=', f_loss)#1D?

rho=[]
for i in range(len(f_loss)):
    rho.append(f_loss[i]/f_loss[0])
pt.plot(rho, '-b')
pt.show()
