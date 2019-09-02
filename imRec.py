# Lib imports
# import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Read .mat file
trainMat = sio.loadmat('data/train_32x32.mat')

# Print parts of trainMat
# print('Train X: ', trainMat['X'])
# print('Train y: ', trainMat['y'])
print('Train X1: ', trainMat['X'][:, :, :, 0])
print('Train y1: ', trainMat['y'][0])

# Print shapes of trainMat
print('Train X_shape', trainMat['X'].shape)
print('Train X1_shape', trainMat['X'][:, :, :, 0].shape)
print('Train y_shape', trainMat['y'].shape)
print('Train y1_shape', trainMat['y'][0].shape)

# Show images
plt.imshow(trainMat['X'][:, :, :, 0])
plt.show()
