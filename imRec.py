# Lib imports
# import numpy as np
import scipy.io as sio
# import matplotlib.pyplot as plt

# Module imports
from imPP import imagePP
from imNN import compileNN, trainNN, testNN

# Read .mat file
trainMat = sio.loadmat('data/train_32x32.mat')

# Print parts of trainMat
# print('Train X: ', trainMat['X'])
# print('Train y: ', trainMat['y'])
# print('Train X1: ', trainMat['X'][:, :, :, 0])
# print('Train y1: ', trainMat['y'][0])

# Print shapes of trainMat
# print('Train X_shape', trainMat['X'].shape)
# print('Train X1_shape', trainMat['X'][:, :, :, 0].shape)
# print('Train y_shape', trainMat['y'].shape)
# print('Train y1_shape', trainMat['y'][0].shape)

# Send trainMat to imagePP for pre-processing
data = imagePP(trainMat)

# # Send data for dimensions to compile NN
# compileNN(data)

# # Send training data with labels to NN
# data, labels = trainNN()

# # Send test data with labels to NN
# data, labels = testNN()
