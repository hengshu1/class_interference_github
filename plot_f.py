import matplotlib.pyplot as pt
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--model', default='vgg19', type=str, help='model name')
args = parser.parse_args()

from main import classes

# f_loss = np.load('results/f_vgg_sgd_alpha_'+str(args.lr)+'.npy')
# fc_loss = np.load('results/fc_vgg_sgd_alpha_'+str(0.001)+'.npy')
# print('f_loss.shape=', f_loss.shape)
# print('fc_loss0001.shape=', fc_loss.shape)

file_name = 'results/fc_' + args.model + 'lrmode_schedule_sgd_alpha_0.1.npy'
file_gmatrix_name = 'results/Gmatrix_' + args.model + 'lrmode_schedule_sgd_alpha_0.1.npy'

print('file=', file_name)
fc_loss= np.load(file_name)


gmat_loss = np.load(file_gmatrix_name)
print(gmat_loss.shape)
fc_loss_from_diag = np.diagonal(gmat_loss, axis1=1, axis2=2)

fc_loss = fc_loss_from_diag

c1, c2 = 3, 5

# pt.plot(f_loss, '--m.', label='loss--gross')
# pt.plot(fc_loss[:, -1], '-bo', label='loss--'+classes[-1])
# pt.plot(fc_loss[:, 1], '--g+', label='loss--'+classes[1])
# pt.plot(fc_loss[:, c1], '-k+', label='loss--'+classes[c1])
# pt.plot(fc_loss[:, c2], '-r.', label='loss--'+classes[c2])

# pt.plot(fc_loss[:, 0], ':bs', label='loss--'+classes[0])
# pt.plot(fc_loss[:, 4], ':gp', label='loss--'+classes[4])
# pt.plot(fc_loss[:, 2], ':r<', label='loss--'+classes[2])
# pt.plot(fc_loss[:, 6], ':c+', label='loss--'+classes[6])
# pt.plot(fc_loss[:, 7], ':yx', label='loss--'+classes[7])
# pt.plot(fc_loss[:, 8], ':mo', label='loss--'+classes[8])
# pt.legend(fontsize=18, labelcolor='linecolor')
# pt.xlabel('epochs', fontsize=20)
# pt.ylabel('loss', fontsize=20)
# # pt.xlim([-5, 175])
# pt.xlim([-5, 160])
# pt.ylim([20, 105])


# pt.figure()
# pt.plot(gmat_loss[:, c1, c2], '-k+', label='--% of {}s predicted as {}'.format(classes[c1].lower(), classes[c2].upper()))
# # pt.plot(gmat_loss[:, c1, c2+1], '--r+', label='--% of {}s predicted as {}'.format(classes[c1].lower(), classes[c2+1].upper()))
# # pt.plot(gmat_loss[:, c2, c1], '--r.', label='--% of {}s predicted as {}'.format(classes[c2].lower(), classes[c1].upper()))
# pt.legend(fontsize=18, labelcolor='linecolor')
# pt.xlabel('epochs', fontsize=20)
# pt.xlim([-5, 160])
# pt.show()

pt.figure()

pt.plot(gmat_loss[:, c1, c1], '-k', label='recall rate--'+classes[c1])
pt.plot(gmat_loss[:, c2, c2], '--c', label='recall rate--'+classes[c2])

c1 = 3
markers=['--k+', ':ko', '-.ks',
         '-bv', '--b<', '-rx',
         '-y*', '--rp', ':c1',
         ':gh'
         ]
for c2 in range(len(classes)):
    if c2 != c1:
        pt.plot(gmat_loss[:, c1, c2], markers[c2], label='--% of {}s predicted as {}'.format(classes[c1].lower(), classes[c2].upper()))
# pt.plot(gmat_loss[:, c1, c1], '-m', label='--% of {}s predicted as {}'.format(classes[c1].lower(), classes[c1].upper()))
# pt.plot(gmat_loss[:, c1, 5], '-ro', label='--% of {}s predicted as {}'.format(classes[c1].lower(), classes[5].upper()))
# pt.plot(gmat_loss[:, c1, 1], '-.k+', label='--% of {}s predicted as {}'.format(classes[c1].lower(), classes[1].upper()))
# pt.plot(gmat_loss[:, c1, 9], '--b', label='--% of {}s predicted as {}'.format(classes[c1].lower(), classes[9].upper()))
# pt.plot(gmat_loss[:, c1, 7], ':gx', label='--% of {}s predicted as {}'.format(classes[c1].lower(), classes[7].upper()))
pt.legend(fontsize=14, labelcolor='linecolor')
pt.xlabel('epochs', fontsize=15)
pt.xlim([-5, 160])
pt.show()


