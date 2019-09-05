# Lib imports
# import numpy as np
import scipy.io as sio

# Module imports
from CNN import test

# Read testfiles
testMat = sio.loadmat('data/test_32x32.mat')

images = testMat['X']
labels = testMat['y']

predictions = test(images, labels)
