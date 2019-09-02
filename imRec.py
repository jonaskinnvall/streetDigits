# Lib imports
# import numpy as np
import scipy.io as sio

# Read .mat file
trainMat = sio.loadmat('data/train_32x32.mat')

# Print parts of trainMat
print('Train X: ', trainMat['X'])
print('Train y: ', trainMat['y'])
print('Train X1: ', trainMat['X'][0])
print('Train y1: ', trainMat['y'][0])

print('Train X_shape', trainMat['X'].shape)
print('Train y_shape', trainMat['y'].shape)
