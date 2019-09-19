# Lib imports
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# Module imports
from CNN import evaluate, predict

# Read testfiles
testMat = sio.loadmat('data/test_32x32.mat')

images = testMat['X']
labels = testMat['y']

# Change layout of dimensions to fit CNN model and normalize values
images = np.moveaxis(images, -1, 0)
images = images.astype('float32')
images /= 255.0

# Change 10s to 0s and change labels to binary class matrix
labels = np.where(labels == 10, 0, labels)
labels = labels.astype('float32')
# labels /= 10.0
labels = to_categorical(labels)

# Let user choose if they want to evaluate CNN or use it to predict digits
print("Do you want to evaluate CNN or predict digits with CNN?")
response = None
while response not in {"e", "p"}:
    response = input("Please enter 'e' or 'p': ")

# Call evaluate function
if response == 'e':
    evaluation = evaluate(images, labels)
    # print('TEST LOSS, TEST ACC:', evaluation[0])
    print('TEST LOSS: %.3f, TEST ACC: %.3f%%' %
          (evaluation[0], evaluation[1]*100))

# Call prediction function
elif response == 'p':
    predictions = predict(images)
    digits = np.argmax(predictions, axis=1)
    true = np.argmax(labels, axis=1)
    print('dig', digits.shape)
    print('\n Conf Mat \n =============== \n',
          confusion_matrix(true, digits))
    print('\n Class Rep \n =============== \n',
          classification_report(true, digits))
    # plt.figure(1)
    # plt.subplot(2, 1, 1)
    # plt.hist(digits, bins=10)
    # plt.subplot(2, 1, 2)
    # plt.hist(true, bins=10)
    # plt.show()
