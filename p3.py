import json
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Lambda
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.regularizers import l2
import csv
import os
import numpy as np
import cv2
import math
from random import shuffle
from keras.models import model_from_json
import json
import random

new_model = False

col, row, ch = 160, 80, 3
learning_rate = 0.0001
w_reg=0.00
angleAdjustFactor = 0.75

batch_size = 100
num_epoch = 7


def readLogFiles():
    with open(os.path.join('data/', 'driving_log.csv'), 'r') as f:
        reader = csv.reader(f)
        driving_log_list = list(reader)
        num_features = len(driving_log_list)
        print("Found {} line.".format(num_features))
        logEntries = [("", 0.0, 0) for x in range(num_features*3)]
        for i in range(num_features):
            if(float(driving_log_list[i][4]) > .25 ):
                #center
                logEntries[i*3] = (driving_log_list[i][0].lstrip(),
                      float(driving_log_list[i][3]),
                      0)

                # #all flipped center images
                # logEntries[(i*4)+1] = (driving_log_list[i][0].lstrip(),
                #       float(driving_log_list[i][3]),
                #       1)

                #left
                logEntries[(i*3)+1] = (driving_log_list[i][1].lstrip(),
                      float(driving_log_list[i][3]) + abs(float(driving_log_list[i][3])*angleAdjustFactor),
                      0)
                #right
                logEntries[(i*3)+2] = (driving_log_list[i][2].lstrip(),
                      float(driving_log_list[i][3]) - abs(float(driving_log_list[i][3])*angleAdjustFactor),
                      0)
    return logEntries

def flip(image):
    flipped_image = cv2.flip(image, 1)
    flipped_image = flipped_image[np.newaxis, ...]
    return flipped_image;

def processImage(filename, angle, flipFlag=0):
    angle = angle
    image = cv2.imread(os.path.join('data/', filename))
    image = cv2.resize(image, (col, row))
    if flipFlag == 1:
        image = flip(image)
        angle = -1.0 * angle
    return image, angle

def getImageGenerator(X_train, batch_size):
    index = 0
    while 1:
        batch_images = np.ndarray(shape=(batch_size, row, col, ch), dtype=float)
        batch_angles = np.ndarray(shape=(batch_size), dtype=float)
        for i in range(batch_size):
            if index >= len(X_train):
                index = 0
                shuffle(X_train)

            filename = X_train[index][0]
            angle = X_train[index][1]
            flipFlag = X_train[index][2]

            final_image, angle = processImage(filename, angle, flipFlag)

            final_angle = np.ndarray(shape=(1), dtype=float)
            final_angle[0] = angle

            batch_images[i] = final_image
            batch_angles[i] = final_angle
            index += 1
        yield batch_images, batch_angles

def getModel():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))

    model.add(Convolution2D(nb_filter=24,
                        nb_row=5,
                        nb_col=5,
                        subsample=(2,2),
                        border_mode='valid',
                        input_shape=(row, col, ch)))
    model.add(Activation('relu'))

    # CNN Layer 2
    model.add(Convolution2D(nb_filter=36,
                        nb_row=5,
                        nb_col=5,
                        subsample=(2,2),
                        border_mode='valid'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    # CNN Layer 3
    model.add(Convolution2D(nb_filter=48,
                        nb_row=5,
                        nb_col=5,
                        subsample=(2,2),
                        border_mode='valid'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    # CNN Layer 4
    model.add(Convolution2D(nb_filter=64,
                        nb_row=3,
                        nb_col=3,
                        subsample=(1,1),
                        border_mode='valid'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    # CNN Layer 5
    model.add(Convolution2D(nb_filter=64,
                        nb_row=3,
                        nb_col=3,
                        subsample=(1,1),
                        border_mode='valid'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    # Flatten
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # FCNN Layer 1
    model.add(Dense(512, input_shape=(2496,), name="hidden1", W_regularizer=l2(w_reg)))
    model.add(Activation('relu'))

    # FCNN Layer 2
    model.add(Dense(256, name="hidden2", W_regularizer=l2(w_reg)))
    model.add(Activation('relu'))

    # FCNN Layer 2
    model.add(Dense(50, name="hidden3", W_regularizer=l2(w_reg)))
    model.add(Activation('relu'))

    # FCNN Layer 3
    model.add(Dense(1, name="output", W_regularizer=l2(w_reg)))

    model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=learning_rate)
              )

    return model

# remove angles that are larger that .98 or lesser than -.98
# remove 0 angles with a probability of 30%
def removeExtremeEntries(logEntries):
    num = 0
    indexToRemove = []

    for i in range(len(logEntries)):
        angle = float(logEntries[i][1])
        if math.isclose(angle, 0, abs_tol=0.001):
            indexToRemove.append(i)
            num = num+1
            continue

        if angle > 0.98 or angle < -0.98:
            indexToRemove.append(i)
            num = num+1
            continue

    logEntries = np.delete(logEntries, indexToRemove, 0)
    return logEntries.tolist(), num

def flipAllImages50Prob(logEntries):
    for i in range(len(logEntries)):
        if np.random.choice([True, False]):
            logEntries.append([logEntries[i][0], logEntries[i][1], 1])
    return logEntries


def flipAllNonZeroAngledImages(logEntries):
    for i in range(len(logEntries)):
        if(logEntries[i][1] != 0.0):
            logEntries.append([logEntries[i][0], logEntries[i][1], 1])
    return logEntries

def biasedCoin(p):
    return True if random.random() < p else False

def addExtraImagesOfLargeAnglesWithFlip(logEntries):
    for i in range(len(logEntries)):
        angle = float(logEntries[i][1])
        if angle > 0.40 and angle < 0.70:
            logEntries.append([logEntries[i][0], logEntries[i][1], 1])
            if biasedCoin(.40):
                logEntries.append([logEntries[i][0], logEntries[i][1], 0])
        if angle < -0.40 and angle > -0.70:
            logEntries.append([logEntries[i][0], logEntries[i][1], 0])
            if biasedCoin(.30):
                logEntries.append([logEntries[i][0], logEntries[i][1], 1])
    return logEntries

def randomlyRemoveZeroAngledImages(logEntries):
    num = 0
    indexToRemove = []
    for i in range(len(logEntries)):
        angle = float(logEntries[i][1])
        if math.isclose(angle, 0, abs_tol=0.001):
            if np.random.choice([True, False]):
                indexToRemove.append(i)
                num = num+1
    logEntries = np.delete(logEntries, indexToRemove, 0)
    return logEntries.tolist(), num

def removeNoneName(X_train):
    num = 0
    indexToRemove = []
    for i in range(len(X_train)):
        filename = X_train[i][0]

        if filename == "":
            indexToRemove.append(i)
            num = num + 1
    X_train = np.delete(X_train, indexToRemove, 0)
    return X_train.tolist(), num

def removeLowAngledImages(logEntries):
    num = 0
    indexToRemove = []
    for i in range(len(logEntries)):
        if(abs(float(logEntries[i][1])) < 0.01):
            indexToRemove.append(i)
            num = num+1
    logEntries = np.delete(logEntries, indexToRemove, 0)
    return logEntries.tolist(), num

def getSamples(array_size, batch_size):
    num_batches = array_size / batch_size
    samples_per_epoch = math.ceil((num_batches / batch_size) * batch_size)
    return samples_per_epoch * batch_size

def custom_train_test_valid(logEntries):
    num_features = len(logEntries)
    shuffle(logEntries)
    num_train_elements = int((num_features/4.)*3.)
    num_valid_elements = int(((num_features/4.)*1.) / 2.)
    X_valid = logEntries[num_train_elements:num_train_elements + num_valid_elements]
    X_test = logEntries[num_train_elements + num_valid_elements:]
    X_train = logEntries[:num_train_elements]

    X_train, noneTrain = removeNoneName(X_train)
    X_test, noneTest = removeNoneName(X_test)
    X_valid, noneValid = removeNoneName(X_valid)

    print("X_train has {} elements. removed {}".format(len(X_train), noneTrain))
    print("X_valid has {} elements. removed {}".format(len(X_valid), noneTest))
    print("X_test has {} elements. remove {}".format(len(X_test), noneValid))
    return X_train, X_test, X_valid

def newModel(logEntries, numEpoch):
    logEntries, numOfZeroAngle = randomlyRemoveZeroAngledImages(logEntries)
    print("Num of randomly removed 0.0 angle -  {}".format(numOfZeroAngle))

    num_features = len(logEntries)
    print("Total number of features after -  {}".format(num_features))

    logEntries = flipAllNonZeroAngledImages(logEntries)
    num_features = len(logEntries)
    print("Number of features after adding flipped non zero angled images - {}".format(num_features))

    X_train, X_test, X_valid = custom_train_test_valid(logEntries)

    new_model = getModel()

    print("Using generator")

    print("starting model")
    history = new_model.fit_generator(
                        getImageGenerator(X_train, batch_size),
                        samples_per_epoch=getSamples(len(X_train), batch_size),
                        max_q_size=10,
                        nb_epoch=numEpoch,
                        verbose=1,
                        validation_data=getImageGenerator(X_valid, batch_size),
                        nb_val_samples=getSamples(len(X_valid), batch_size))

    # Evaluate the accuracy of the model using the test set
    score = new_model.evaluate_generator(
                        generator=getImageGenerator(X_test, batch_size),
                        val_samples=getSamples(len(X_test), batch_size)
                        )
    print("Test score {}".format(score))
    return new_model

def refineModel(model, numEpoch):
    for i in range(8):
        logEntries = readLogFiles()
        print("Number of features read from file -  {}".format(len(logEntries)))

        logEntries, num = removeExtremeEntries(logEntries)
        logEntries, numOfZeroAngle = randomlyRemoveZeroAngledImages(logEntries)
        print("Num of randomly removed 0.0 angle -  {}".format(numOfZeroAngle))

        num_features = len(logEntries)
        print("Total number of features after -  {}".format(num_features))

        logEntries = flipAllImages50Prob(logEntries)
        num_features = len(logEntries)
        print("Number of features after adding flipped non zero angled images - {}".format(num_features))

        X_train, X_test, X_valid = custom_train_test_valid(logEntries)

        print("Using generator")

        print("starting model")
        history = model.fit_generator(
                            getImageGenerator(X_train, batch_size),
                            samples_per_epoch=getSamples(len(X_train), batch_size),
                            max_q_size=10,
                            nb_epoch=numEpoch,
                            verbose=1,
                            validation_data=getImageGenerator(X_valid, batch_size),
                            nb_val_samples=getSamples(len(X_valid), batch_size))

    # Evaluate the accuracy of the model using the test set
    score = model.evaluate_generator(
                        generator=getImageGenerator(X_test, batch_size),
                        val_samples=getSamples(len(X_test), batch_size)
                        )
    print("Test score {}".format(score))
    print(model.summary())

    return model

def saveModel(model, model_fname, model_wfname):
    model_json = model.to_json()
    with open(model_fname, "w") as json_file:
        json.dump(model_json, json_file)
    model.save_weights(model_wfname)
    print("Saved model to disk")

if __name__ == '__main__':
    print("Learning rate - {}.".format(learning_rate))
    print("Number of epochs - {}.".format(num_epoch))
    print("Adjust left and right angle by a factor of - {}".format(angleAdjustFactor))

    if new_model:
        logEntries = readLogFiles()
        print("Number of features read from file -  {}".format(len(logEntries)))

        model = newModel(logEntries, num_epoch)
        saveModel(model, "./model.json", "./model.h5")
    else:
        print("Running in refine mode")
        with open("model_base.json", 'r') as file:
            model = model_from_json(json.load(file))

        model.compile(loss='mean_squared_error',
          optimizer=Adam(lr=0.00001)
          )
        model.load_weights("model_base.h5")
        model = refineModel(model, 1)
        saveModel(model, "./model_refine_4a.json", "./model_refine_4a.h5")
