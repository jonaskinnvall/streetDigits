# Lib imports
import numpy as np
from keras.models import Model, load_model
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D)


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

    # Print summary of CNN model
    cnn.summary()

    # Return model
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


# Evaluate CNN function
def evaluate(data, labels):
    # Load CNN model
    cnn = load_model('./CNN.h5')

    evaluation = cnn.evaluate(data, labels, batch_size=1024)

    return evaluation


# Predict digits with CNN
def predict(data):
    # Load CNN
    cnn = load_model('./CNN.h5')

    predictions = cnn.predict(data, batch_size=1024)

    return predictions
