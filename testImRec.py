# Lib imports
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.utils import to_categorical

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
labels = to_categorical(labels)

# Let user choose if they want to evaluate CNN or use it to predict digits
print("Do you want to evaluate CNN or predict digits with CNN?")
response = None
while response not in {"e", "p"}:
    response = input("Please enter 'e' or 'p': ")

# Call evaluate function
if response == 'e':
    evaluation = evaluate(images, labels)
    print('Eval loss:', evaluation)
    # print summary
    # print('Accuracy: mean=%.3f std=%.3f, n=%d' %
    #       (np.mean(acc) * 100, np.std(acc) * 100,
    #        len(acc)))

    # # box and whisker plots of results
    # plt.boxplot(acc)
    # plt.show()

# Call prediction function
elif response == 'p':
    predictions = predict(images)
    print('Preds', predictions.shape)
