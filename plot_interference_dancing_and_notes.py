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


all_CCTM = np.load(file_gmatrix_name)
print(np.max(all_CCTM))

inter_threshold = 0.1#5.#3.#1.#0.1

inter_class = -2 * np.ones((all_CCTM.shape[0], all_CCTM.shape[1])) #interference class index
for epoch in range(all_CCTM.shape[0]):
    CCTM = all_CCTM[epoch, :, :]
    for i in range(CCTM.shape[0]):
        inter_of_i = -2
        max_this_row = -2
        for j in range(CCTM.shape[1]):
            if i != j and CCTM[i, j] > max_this_row:
                inter_of_i = j
                max_this_row = CCTM[i, j]
        if max_this_row > inter_threshold: # percent
            inter_class[epoch, i] = inter_of_i
print(inter_class[:, 3])

#Fig 7
pt.figure()
pt.plot(inter_class[:, 3], '-r', marker='x', markersize=5, label="CAT(3) notes")
pt.plot(inter_class[:, 5], '-.m', marker=r'$\rho$', markersize=5, label="DOG(5) notes")
pt.plot(inter_class[:, 1], ':b', marker='D', markersize=5, label="CAR(1) notes")
pt.plot(inter_class[:, 9], '--k', marker ='.', markersize=12, label="TRUCK(9) notes")
pt.legend(fontsize=12, labelcolor='linecolor', loc='lower left', ncol=2)
pt.xlabel('epochs', fontsize=12)
pt.xlim([-1, 183])
pt.ylabel('Class that dominates interference', fontsize=12)
pt.show()

#count the times for each dorminating class for the CAT notes
def count_dorminating_times(inter_class, c, c_inter):
    # print('c=', c)
    # print('c_inter=', c_inter)
    # print(inter_class[:, c])
    result = np.where(inter_class[:, c] == c_inter)
    # print('result[0].shape=', result[0].shape)
    # print('result[0].shape[0]=', result[0].shape[0])
    # print('result[1].shape=', result[1].shape)
    return result[0].shape[0]

Gmat_from_train=np.zeros((len(classes), len(classes)))
for c in range(len(classes)):
    for i in range(len(classes)):
        Gmat_from_train[c, i] = count_dorminating_times(inter_class, c=c, c_inter=i)
        print('number of times {} dorminating the interference to {}:{}'.format(classes[i].upper(), classes[3].upper(), Gmat_from_train[c, i]))

print(Gmat_from_train[3, :])#CAT
print(Gmat_from_train[5, :])#DOG
print(Gmat_from_train[8, :])#SHIP
print(Gmat_from_train[1, :])#CAR
print(Gmat_from_train[9, :])#TRUCK
pt.figure()
pt.imshow(Gmat_from_train, cmap = 'hot_r', interpolation='none')
pt.show()

sys.exit(1)

fc_loss_from_diag = np.diagonal(gmat_loss, axis1=1, axis2=2)

fc_loss = fc_loss_from_diag
print('fc_loss_from_diag.shape=', fc_loss_from_diag.shape)


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

#Fig 6
pt.figure()

#check two methods of computing the recall rates for cats are the same? Yes.
# pt.plot(fc_loss[:, 3], '-mo', label='diag cat recall')

pt.plot(gmat_loss[:, c1, c1], '-k', label='recall rate--'+classes[c1])
pt.plot(gmat_loss[:, c2, c2], '--c', label='recall rate--'+classes[c2])

c1 = 3
markers=['--k+', ':ko', '-.ks',
         '-bv', '--b<', '-rx',
         '-y*', '--rp', ':c1',
         ':gh'
         ]
for c2 in range(len(classes)):
    # if c2 != c1:
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


