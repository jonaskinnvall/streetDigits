# Lib imports
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from os import path
# Module imports
from CNN import compile, train

# Read .mat file
trainMat = sio.loadmat('data/train_32x32.mat')

images = trainMat['X']
labels = trainMat['y']

# Change layout of dimensions to fit CNN model and normalize values
images = np.moveaxis(images, -1, 0)
images = images.astype('float32')
images /= 255.0

# Set 10s to 0s and change labels to binary class matrix
labels = np.where(labels == 10, 0, labels)
labels = labels.astype('float32')
# labels /= 10.0
# labels = to_categorical(labels)

# Split training set into training set and validation set
trainIm, valIm, trainL, valL = train_test_split(images,
                                                labels, test_size=0.2,
                                                random_state=42)

# If CNN model doesn't exist send data to compile NN
if not path.exists('./models/CNNspars2.h5'):
    CNNmodel = compile(trainIm[0, :, :, :])
    CNNmodel.save('./models/CNNsparse2.h5')
    CNNmodel.summary()
    print('MODEL COMPILED AND SAVED!')

# Send training data with labels to NN
CNNmodel, history = train(trainIm, trainL, valIm, valL)

CNNmodel.save('./models/CNNsparse2.h5')
print('MODEL TRAINED AND SAVED!')

# Plot training & validation loss values
plt.subplot(2, 1, 1)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation accuracy values
plt.figure(1)
plt.subplot(2, 1, 2)
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
