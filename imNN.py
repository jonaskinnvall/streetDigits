# Lib imports
import numpy as np
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D)
from keras.models import Model
from tensorflow.contrib.keras.api.keras import activations
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.keras.api.keras import metrics


def compileNN(images):

    print('image shape', images.shape)
    # images = np.swapaxes(images, 0, 3)
    input_shape = [images.shape[0], images.shape[1], images.shape[2]]
    print('input dims', input_shape)

    input = Input(shape=input_shape)
    conv = Conv2D(36, (3, 3), activation='relu')(input)
    pool = MaxPooling2D()(conv)
    flat = Flatten()(pool)
    dense = Dense(250, activation='relu')(flat)
    dense = Dense(100, activation='relu')(dense)
    dense = Dense(10, activation='softmax')(dense)

    # Create model
    convModel = Model(inputs=input, outputs=dense)
    convModel.summary()

    # Compile model
    convModel.compile('SGD', loss='categorical_crossentropy',
                      metrics=['accuracy'])

    # Return model to SAVE it
    return convModel


def trainNN(data, labels):
    return data, labels


def testNN(data, labels):
    return data, labels
