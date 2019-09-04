# Lib imports
# import numpy as np
import scipy.io as sio

# Module imports
from imNN import compileNN, trainNN

# Read .mat file
trainMat = sio.loadmat('data/train_32x32.mat')
images = trainMat['X']
labels = trainMat['y']

# Send data for dimensions to compile NN, returns model
model = compileNN(images)
# Save model
model.save('model.h5')

# Send training data with labels to NN
# data, labels = trainNN(images, labels)
