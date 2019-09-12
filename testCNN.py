# Lib imports
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc, roc_auc_score

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
# labels = to_categorical(labels)

# Let user choose if they want to evaluate CNN or use it to predict digits
# print("Do you want to evaluate CNN or predict digits with CNN?")
# response = None
# while response not in {"e", "p"}:
#     response = input("Please enter 'e' or 'p': ")

# Call evaluate function
# if response == 'e':
evaluation = evaluate(images, labels)
# print('TEST LOSS, TEST ACC:', evaluation[0])
print('TEST LOSS: %.3f, TEST ACC: %.3f%%' % (evaluation[0], evaluation[1]*100))

# Call prediction function
# elif response == 'p':
#     predictions = predict(images)
#     print('Preds', predictions.shape)
# print('Preds', predictions)

# Create ROC curve and from that AUC
# auc_score = roc_auc_score(labels, predictions)
# print('AUC', auc_score)
# fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=10)
# auc = auc(fpr, tpr)
# print('AUC:', auc)

# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.show()
