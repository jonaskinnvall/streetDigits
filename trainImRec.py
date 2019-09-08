# Lib imports
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Module imports
from CNN import compile, train

# Read .mat file
trainMat = sio.loadmat('data/train_32x32.mat')

images = trainMat['X']
images = np.moveaxis(images, -1, 0)
labels = trainMat['y']

# Normalize image values and change labels to binary class matrix
images = images.astype('float32')
images /= 255.0

# Set 10s to 0s
labels = np.where(labels == 10, 0, labels)
print(np.min(labels))
print(np.max(labels))
# FIX THIS!!!!
print("l un", np.unique(labels))
labels = to_categorical(labels)
print("l sha2", labels.shape)
print("l un", np.unique(labels))

# Split training set into training set and validation set
trainIm, valIm, trainL, valL = train_test_split(images,
                                                labels, test_size=0.2,
                                                random_state=42)

# Send data for dimensions to compile NN, returns model
CNNmodel = compile(trainIm[0, :, :, :])

# Save model
CNNmodel.save('CNN.h5')

# Send training data with labels to NN
CNNmodel, history = train(trainIm, trainL, valIm, valL)

CNNmodel.save('CNN.h5')

# Plot training & validation accuracy values
plt.figure(1)
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.figure(2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
