import matplotlib.pyplot as pt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
args = parser.parse_args()

from main import classes

f_loss = np.load('results/f_vgg_sgd_alpha_'+str(args.lr)+'.npy')
fc_loss = np.load('results/fc_vgg_sgd_alpha_'+str(0.001)+'.npy')
print('f_loss.shape=', f_loss.shape)
print('fc_loss0001.shape=', fc_loss.shape)


pt.plot(f_loss, '--m.', label='loss--gross')
pt.plot(fc_loss[:, -1], '--b', label='loss--'+classes[-1])
pt.plot(fc_loss[:, 1], '-.g+', label='loss--'+classes[1])
pt.plot(fc_loss[:, 3], '-k', label='loss--'+classes[3])
pt.plot(fc_loss[:, 5], '--r*', label='loss--'+classes[5])

# pt.plot(fc_loss[:, 0], ':bs', label='loss--'+classes[0])
# pt.plot(fc_loss[:, 4], ':gp', label='loss--'+classes[4])
# pt.plot(fc_loss[:, 2], ':r<', label='loss--'+classes[2])
# pt.plot(fc_loss[:, 6], ':c+', label='loss--'+classes[6])
# pt.plot(fc_loss[:, 7], ':yx', label='loss--'+classes[7])
# pt.plot(fc_loss[:, 8], ':mo', label='loss--'+classes[8])
pt.legend(fontsize=18, labelcolor='linecolor')
pt.xlabel('epochs', fontsize=20)
pt.ylabel('loss', fontsize=20)
pt.xlim([-5, 105])
pt.show()
