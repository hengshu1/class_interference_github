import numpy as np
import matplotlib.pylab as plt

file = model_path = 'results/model_vgg_sgd_alpha_'+str(0.01) + '_cat_dog_egomodels_loss.npy'
losses = np.load(file)

plt.figure(1)
print(losses.shape)
plt.imshow(losses, interpolation='none')

hf = plt.figure(2)
ha = hf.add_subplot(111, projection='3d')
x = range(losses.shape[0])
y = range(losses.shape[1])
X, Y = np.meshgrid(x, y)
ha.plot_surface(X, Y, losses)

plt.show()

# def f(x, y):
#     return np.sin(np.sqrt(x ** 2 + y ** 2))
#
# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)
#
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)
#
# print(X.shape)
# print(Y.shape)
# print(Z.shape)
# print(X[0, :])
# print(X[1, :])#the same
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z');
