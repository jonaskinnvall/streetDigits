# Description

Detecting single digits in 32x32 images from Format 2 The Street View House Numbers (SVHN).

## Method

Using Keras with Tensorflow backend to create a Convolutional Neural Network (CNN) which
learns to detect the MNIST-like 32x32 image SVHN data set.

Testing is only done through evaluation which prints the loss and accuracy of the model on
test data it has not seen before.
Predictions are commented out, along with the code for users to choose through input if they
want to evaluate the model or use it for prediction.

Predictions would be nice with a ROC curve together with its AUC to show performance.
But to do this a new model has to be compiled using Keras Sequential model with the
Keras-Classifier wrapper for sklearnto be able to use ROC and AUC from sklearn.

## Models

No changes needed in the code to use "bestSparse" model which uses sparse categorical crossentropy.

To use the "bestCheck" model, which uses categorical crossentropy as loss and metric, the function
"to_categorical" has to be used to change the labels to a binary class matrix. To_categorical is
commented out in both train and test file right after 10s are changed to 0s in labels arrays.

### Link to data set

http://ufldl.stanford.edu/housenumbers/
