# Lib imports
# import numpy as np
from keras.models import Model, load_model
from keras.layers import (Input, Conv2D, Dense, Flatten,
                          MaxPooling2D, Dropout)


# Compile CNN function
def compile(image):

    # print(image.shape)

    input = Input(shape=image.shape)
    conv = Conv2D(32, (4, 4), padding='same', activation='relu')(input)
    pool = MaxPooling2D()(conv)
    conv2 = Conv2D(64, (4, 4), padding='valid', activation='relu')(pool)
    pool2 = MaxPooling2D()(conv2)
    flat = Flatten()(pool2)
    dense = Dense(1024, activation='relu')(flat)
    dense2 = Dense(256, activation='relu')(dense)
    dense3 = Dense(64, activation='relu')(dense2)
    drop = Dropout(0.3)(dense3)
    dense3 = Dense(10, activation='softmax')(drop)

    # Create model
    cnn = Model(inputs=input, outputs=dense3)

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
    cnn = load_model('CNN2.h5')

    # Train CNN model
    n_epochs = 600
    b_size = 256
    history = cnn.fit(data, labels, batch_size=b_size,
                      epochs=n_epochs, verbose=1,
                      validation_data=(valData, valLabels)).history

    return cnn, history


# Evaluate CNN function
def evaluate(data, labels):
    # Load CNN model
    cnn = load_model('CNN.h5')

    evaluation = cnn.evaluate(data, labels, batch_size=1024)

    return evaluation


# Predict digits with CNN
def predict(data):
    # Load CNN
    cnn = load_model('CNN.h5')

    predictions = cnn.predict(data, batch_size=1024)

    return predictions
