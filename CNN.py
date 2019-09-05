# Lib imports
import numpy as np
from keras.layers import (Input, Conv2D, Dense, Flatten,
                          MaxPooling2D, Dropout, BatchNormalization)
from keras.models import Model, load_model


# Compile CNN function
def compile(image):

    # input_shape = image.shape

    input = Input(shape=image.shape)
    conv = Conv2D(64, (3, 3), activation='relu')(input)
    pool = MaxPooling2D()(conv)
    flat = Flatten()(pool)
    dense = Dense(100, activation='relu')(flat)
    dense = Dense(10, activation='softmax')(dense)
    # dense = Dense(1, activation='softmax')(dense)

    # Create model
    cnn = Model(inputs=input, outputs=dense)

    # Compile model
    cnn.compile('SGD', loss='categorical_crossentropy',
                metrics=['accuracy'])

    cnn.summary()

    # Return model to SAVE it
    return cnn


# Train CNN function
def train(data, labels, valData, valLabels):
    # Load CNN model
    cnn = load_model('./CNN.h5')

    # Train CNN model
    n_epochs = 5
    b_size = 1024
    history = cnn.fit(data, labels, batch_size=b_size,
                      epochs=n_epochs, verbose=1,
                      validation_data=(valData, valLabels)).history

    return cnn, history


# Test CNN function
def test(data, labels):
    return data, labels
