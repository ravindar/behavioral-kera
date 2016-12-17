import json
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Lambda
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import csv
import os
import numpy as np
import cv2

col, row, ch = 160, 80, 3  # camera format
learning_rate = 0.0001
dropout_factor = 0.3
w_reg=0.00

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))

    model.add(Convolution2D(nb_filter=24,
                        nb_row=5,
                        nb_col=5,
                        subsample=(2,2),
                        border_mode='valid',
                        input_shape=(row, col, ch),
                        W_regularizer=l2(w_reg)))
    model.add(Activation('relu'))
    # CNN Layer 2
    model.add(Convolution2D(nb_filter=36,
                        nb_row=5,
                        nb_col=5,
                        subsample=(2,2),
                        border_mode='valid',
                        W_regularizer=l2(w_reg)))
    model.add(Dropout(dropout_factor))
    model.add(Activation('relu'))
    # CNN Layer 3
    model.add(Convolution2D(nb_filter=48,
                        nb_row=5,
                        nb_col=5,
                        subsample=(2,2),
                        border_mode='valid',
                        W_regularizer=l2(w_reg)))
    model.add(Dropout(dropout_factor))
    model.add(Activation('relu'))
    # CNN Layer 4
    model.add(Convolution2D(nb_filter=64,
                        nb_row=3,
                        nb_col=3,
                        subsample=(1,1),
                        border_mode='valid',
                        W_regularizer=l2(w_reg)))
    model.add(Dropout(dropout_factor))
    model.add(Activation('relu'))
    # CNN Layer 5
    model.add(Convolution2D(nb_filter=64,
                        nb_row=3,
                        nb_col=3,
                        subsample=(1,1),
                        border_mode='valid',
                        W_regularizer=l2(w_reg)))
    model.add(Dropout(dropout_factor))
    model.add(Activation('relu'))
    # Flatten
    model.add(Flatten())
    # FCNN Layer 1
    model.add(Dense(512, input_shape=(2496,), name="hidden1", W_regularizer=l2(w_reg)))
    model.add(Activation('relu'))
    # FCNN Layer 2
    model.add(Dense(256, name="hidden2", W_regularizer=l2(w_reg)))
    model.add(Activation('relu'))
    # FCNN Layer 3
    model.add(Dense(1, name="output", W_regularizer=l2(w_reg)))
    model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=learning_rate)
              )

    return model


def readLog():
    with open(os.path.join('data/', 'driving_log.csv'), 'r') as f:
        reader = csv.reader(f)
        driving_log_list = list(reader)
        num_features = len(driving_log_list)
        print("Found {} features.".format(num_features))

        X_train = np.ndarray(shape=(num_features, row, col, ch), dtype=float)
        Y_train = np.ndarray(shape=(num_features), dtype=float)
        for i in range(num_features):
            filename = driving_log_list[i][0].lstrip()
            angle = driving_log_list[i][3]
            image = process_image(filename)
            angle_arr = np.ndarray(shape=(1), dtype=float)
            angle_arr[0] = angle

            X_train[i] = image
            Y_train[i] = angle_arr

        return X_train, Y_train

def process_image(filename):
    image = cv2.imread(os.path.join('data/', filename))
    image = cv2.resize(image, (col, row))
    return image

if __name__ == '__main__':
    X_train, Y_train = readLog()

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_train,
        Y_train,
        test_size=0.15,
        random_state=0)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_train,
        Y_train,
        test_size=0.15,
        random_state=0)

    print("X_train has {} elements.".format(len(X_train)))
    print("X_valid has {} elements.".format(len(X_valid)))
    print("X_test has {} elements.".format(len(X_test)))

    model = get_model()

    # print("Using generator")
    #
    # datagen = ImageDataGenerator(
    #     featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     zca_whitening=True
    # )
    #
    # datagen.fit(X_train)

    print("starting model")
    # history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=50),
    #                     samples_per_epoch=X_train.shape[0],
    #                     nb_epoch=10,
    #                     verbose=1,
    #                     validation_data=(X_valid, Y_valid))

    history = model.fit(X_train, Y_train,
                        batch_size=len(X_train),
                        nb_epoch=6,
                        verbose=1,
                        validation_data=(X_valid, Y_valid),
                        shuffle=True)

    # Evaluate the accuracy of the model using the test set
    score = model.evaluate(X_test, Y_test, batch_size=len(X_test), verbose=1)

    print("Test score {}".format(score))

    ################################################################
    # Save the model and weights
    ################################################################
    model_json = model.to_json()
    with open("./model.json", "w") as json_file:
        json.dump(model_json, json_file)
    model.save_weights("./model.h5")
    print("Saved model to disk")
