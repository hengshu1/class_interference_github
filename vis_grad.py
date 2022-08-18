import matplotlib.pyplot as plt
import numpy as np
import sys


#batch size comparison
model_SB = 'results/model_vgg_sgd_alpha_'+str(0.001)
model_LB = 'results/model_vgg_sgd_alpha_'+str(0.001)+'_batchsize1024'

#learning rate comparison
# model_SB = 'results/model_vgg_sgd_alpha_'+str(0.01)
# model_LB = 'results/model_vgg_sgd_alpha_'+str(0.001)

grad_SB = np.load(model_SB+'_grad.npy')
grad_LB = np.load(model_LB+'_grad.npy')

#take absolute
grad_SB = np.abs(grad_SB)
grad_LB = np.abs(grad_LB)

# print('grad_SB.shape=', grad_SB.shape)
# print('grad_LB.shape=', grad_LB.shape)

bins = [0, 0.001, 0.002, 0.0025, 0.003,
        0.005, 0.0075, 0.009, 0.01, 0.015, 0.02]
counts_LB, bins_LB = np.histogram(
    grad_LB, bins=bins)
# counts_SB, bins_SB = np.histogram(grad_SB)
# counts_SB, bins_SB = np.histogram(grad_SB, bins=10)
counts_SB, bins_SB = np.histogram(grad_SB, bins=bins_LB)
# counts_LB, bins_LB = np.histogram(grad_LB, bins=10)

print('SB:')
print(counts_SB)
print('LB')
print(counts_LB)

# print('counts_SB len', len(counts_SB))
# print('counts_LB len', len(counts_LB))

# print('log LB:')
# print(np.log(counts_SB))
# print('log SB')
# print(np.log(counts_LB))

print('double check bins:')
print('SB')
print(bins_SB)
print('LB')
print(bins_LB)
# print(len(bins_SB))
# print(len(bins_LB))

# plt.plot(counts_SB, '-b+', label='SB')
# plt.plot(counts_LB, '--ro', label='LB')
#ignore the majority near-zero gradients
xticks = np.arange(10)  # bins_SB[2:]
print('len xticks=', len(xticks))
print('counts_SB len=', len(counts_SB))

#seems an exponential decay
# plt.plot(counts_SB[1:], '-b+', label='SB')
# plt.plot(counts_LB[1:], '--ro', label='LB')

plt.bar(xticks, np.log(counts_SB), color='b',
        label='batchsize=128', width=0.25)
plt.bar(xticks+0.3, np.log(counts_LB), color='r',
        label='batchsize=1024', width=0.25)
# plt.bar(xticks, np.log(counts_SB), color='k',
#         label=r'lr=0.01', width=0.25)
# plt.bar(xticks+0.3, np.log(counts_LB), color='m',
#         label=r'lr=0.001', width=0.25)

plt.xticks(xticks, np.array(bins)[1:].tolist())
plt.xlabel('abs(gradient) bins', fontsize=16)
plt.ylabel('log(counts)', fontsize=16)
plt.legend(fontsize=20)
plt.show()


# plt.hist(grad_SB, bins=10)
# plt.hist(grad_LB, bins=10)
# plt.show()
#abs
