# Lib imoprts
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

trainMat = sio.loadmat('data/train_32x32.mat')

print(trainMat['X'].shape)

for i in range(5):
    print('Train y1: ', trainMat['y'][i])
    print("R color:", trainMat['X'][:, :, 0, i].shape)
    print("G color:", trainMat['X'][:, :, 1, i].shape)
    print("B color:", trainMat['X'][:, :, 2, i].shape)

    # Show images
    plt.figure(1)
    plt.subplot(4, 1, 1)
    plt.imshow(trainMat['X'][:, :, :, i])
    plt.subplot(4, 1, 2)
    plt.imshow(trainMat['X'][:, :, 0, i])
    plt.subplot(4, 1, 3)
    plt.imshow(trainMat['X'][:, :, 1, i])
    plt.subplot(4, 1, 4)
    plt.imshow(trainMat['X'][:, :, 2, i])
    plt.show()
