# Lib imports
# import numpy as np
import scipy.io as sio

# Module imports
from imNN import testNN

# Read testfiles
testMat = sio.loadmat('data/test_32x32.mat')

images = testMat['X']
labels = testMat['y']

predictions = testNN(images, labels)
