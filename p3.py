from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Lambda, LeakyReLU
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
import argparse

col, row, ch = 160, 80, 3
w_reg=0.00

def readLogFiles(angleAdjust, location):
    with open(os.path.join(location, 'driving_log.csv'), 'r') as f:
        reader = csv.reader(f)
        driving_log_list = list(reader)
        num_features = len(driving_log_list)
        print("Found {} line.".format(num_features))
        logEntries = [("", 0.0, 0) for x in range(num_features*3)]
        for i in range(num_features):
            # if(float(driving_log_list[i][4]) > .25 ):
            #center
            logEntries[i*3] = (driving_log_list[i][0].lstrip(),
                  float(driving_log_list[i][3]),
                  0)

            #left
            logEntries[(i*3)+1] = (driving_log_list[i][1].lstrip(),
                  float(driving_log_list[i][3]) + abs(float(driving_log_list[i][3])*angleAdjust),
                  0)
            #right
            logEntries[(i*3)+2] = (driving_log_list[i][2].lstrip(),
                  float(driving_log_list[i][3]) - abs(float(driving_log_list[i][3])*angleAdjust),
                  0)
    return logEntries

def flip(image):
    flipped_image = cv2.flip(image, 1)
    flipped_image = flipped_image[np.newaxis, ...]
    return flipped_image;

def processImage(filename, angle, dataLocation, flipFlag=0):
    angle = angle
    image = cv2.imread(os.path.join(dataLocation, filename))
    image = cv2.resize(image, (col, row))
    if flipFlag == 1:
        image = flip(image)
        angle = -1.0 * angle
    return image, angle

def getImageGenerator(X_train, batchSize, dataLocation):
    index = 0
    while 1:
        batchImages = np.ndarray(shape=(batchSize, row, col, ch), dtype=float)
        batchAngles = np.ndarray(shape=(batchSize), dtype=float)
        for i in range(batchSize):
            if index >= len(X_train):
                index = 0
                shuffle(X_train)

            filename = X_train[index][0]
            angle = X_train[index][1]
            flipFlag = X_train[index][2]

            finalImage, angle = processImage(filename, angle, dataLocation, flipFlag)

            finalAngle = np.ndarray(shape=(1), dtype=float)
            finalAngle[0] = angle

            batchImages[i] = finalImage
            batchAngles[i] = finalAngle
            index += 1
        yield batchImages, batchAngles

def getModel(learnRate):
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
    model.add(LeakyReLU(name="leaky relu 1"))

    # CNN Layer 2
    model.add(Convolution2D(nb_filter=36,
                        nb_row=5,
                        nb_col=5,
                        subsample=(2,2),
                        border_mode='valid'))
    model.add(Dropout(0.5))
    model.add(LeakyReLU(name="leaky relu 2"))

    # CNN Layer 3
    model.add(Convolution2D(nb_filter=48,
                        nb_row=5,
                        nb_col=5,
                        subsample=(2,2),
                        border_mode='valid'))
    model.add(Dropout(0.5))
    model.add(LeakyReLU(name="leaky relu 3"))

    # CNN Layer 4
    model.add(Convolution2D(nb_filter=64,
                        nb_row=3,
                        nb_col=3,
                        subsample=(1,1),
                        border_mode='valid'))
    model.add(Dropout(0.5))
    model.add(LeakyReLU(name="leaky relu 4"))

    # CNN Layer 5
    model.add(Convolution2D(nb_filter=64,
                        nb_row=3,
                        nb_col=3,
                        subsample=(1,1),
                        border_mode='valid'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(name="leaky relu 5"))

    # Flatten
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(LeakyReLU(name="leaky relu 6"))

    # FCNN Layer 1
    model.add(Dense(512, input_shape=(2496,), name="hidden1", W_regularizer=l2(w_reg)))
    model.add(LeakyReLU(name="leaky relu 7"))

    # FCNN Layer 2
    model.add(Dense(256, name="hidden2", W_regularizer=l2(w_reg)))
    model.add(LeakyReLU(name="leaky relu 8"))

    # FCNN Layer 2
    model.add(Dense(50, name="hidden3", W_regularizer=l2(w_reg)))
    model.add(LeakyReLU(name="leaky relu 9"))

    # FCNN Layer 3
    model.add(Dense(1, name="output", W_regularizer=l2(w_reg)))

    model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=learnRate)
              )

    return model

# remove abs(angles) less .01
def removeExtremeEntries(logEntries):
    num = 0
    indexToRemove = []

    for i in range(len(logEntries)):
        angle = float(logEntries[i][1])
        if abs(angle) < 0.01:
            if biasedCoin(.8):
                indexToRemove.append(i)
                num = num+1

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

def getSamples(featureSize, batchSize):
    numBatches = featureSize / batchSize
    samplesPerEpoch = math.ceil((numBatches / batchSize) * batchSize)
    return samplesPerEpoch * batchSize

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

def newModel(model, numEpoch, batchSize, adjustAngle, dataLocation):
    for i in range(10):
        logEntries = readLogFiles(adjustAngle, dataLocation)
        print("Number of features read from file -  {}".format(len(logEntries)))

        logEntries, num = removeExtremeEntries(logEntries)
        print("Num of extreme angles removed -  {}".format(num))

        num_features = len(logEntries)
        print("Total number of features after -  {}".format(num_features))

        logEntries = flipAllImages50Prob(logEntries)
        num_features = len(logEntries)
        print("Number of features after flipping images randomly - {}".format(num_features))

        # logEntries = flipAllNonZeroAngledImages(logEntries)
        # num_features = len(logEntries)
        # print("Number of features after adding flipped non zero angled images - {}".format(num_features))

        X_train, X_test, X_valid = custom_train_test_valid(logEntries)

        print("Using generator")

        print("starting model")
        history = model.fit_generator(
                            getImageGenerator(X_train, batchSize, dataLocation),
                            samples_per_epoch=getSamples(len(X_train), batchSize),
                            max_q_size=10,
                            nb_epoch=numEpoch,
                            verbose=1,
                            validation_data=getImageGenerator(X_valid, batchSize, dataLocation),
                            nb_val_samples=getSamples(len(X_valid), batchSize))

        # Evaluate the accuracy of the model using the test set
        score = model.evaluate_generator(
                            generator=getImageGenerator(X_test, batchSize, dataLocation),
                            val_samples=getSamples(len(X_test), batchSize)
                            )
        print("Test score {}".format(score))
    print(model.summary())
    return model

def refineModel(model, numEpoch, batchSize, adjustAngle, dataLocation):
    for i in range(5):
        logEntries = readLogFiles(adjustAngle, dataLocation)
        print("Number of features read from file -  {}".format(len(logEntries)))

        logEntries, num = removeExtremeEntries(logEntries)
        print("Num removed angles below abs(angle) < .01 -  {}".format(num))

        num_features = len(logEntries)
        print("Total number of features after -  {}".format(num_features))

        logEntries = flipAllImages50Prob(logEntries)
        num_features = len(logEntries)
        print("Number of features after adding flipped non zero angled images - {}".format(num_features))

        logEntries = addExtraImagesOfLargeAnglesWithFlip(logEntries)
        num_features = len(logEntries)
        print("After repeating large angled images with flips - {}".format(num_features))

        X_train, X_test, X_valid = custom_train_test_valid(logEntries)

        print("Using generator")

        print("starting model")
        history = model.fit_generator(
                            getImageGenerator(X_train, batchSize, dataLocation),
                            samples_per_epoch=getSamples(len(X_train), batchSize),
                            max_q_size=10,
                            nb_epoch=numEpoch,
                            verbose=1,
                            validation_data=getImageGenerator(X_valid, batchSize, dataLocation),
                            nb_val_samples=getSamples(len(X_valid), batchSize))

        # Evaluate the accuracy of the model using the test set
        score = model.evaluate_generator(
                            generator=getImageGenerator(X_test, batchSize, dataLocation),
                            val_samples=getSamples(len(X_test), batchSize)
                            )
        print("Test score {}".format(score))
    print(model.summary())

    return model

def saveModel(model, modelFileName, modelWeightFName):
    model_json = model.to_json()
    with open(modelFileName, "w") as json_file:
        json.dump(model_json, json_file)
    model.save_weights(modelWeightFName)
    print("Saved model to disk")

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       pass

if __name__ == '__main__':

    # for new model - batch=100, numEpoch=1, learningRate=.0001, adjustAngleBy=.75
    # for refine model - batch=100, numEpoch=1, learningRate=.00001, adjustAngleBy=.75
    parser = argparse.ArgumentParser(description='Remote Driving')

    parser.add_argument('baseModel', type=str, help='Need to pass what mode the model is running as')
    parser.add_argument('learningRate', type=float, help='learning rate.')
    parser.add_argument('dataLocation', type=str, help='location of data.')
    parser.add_argument('saveModelFName', type=str, help='model file name.')

    args = parser.parse_args()

    baseModel = t_or_f(args.baseModel)
    batch = 32
    numEpoch = 1

    lRate = args.learningRate
    angleAdj = .75

    saveModelFName = args.saveModelFName

    # dataLocation='data/' or 'data/more_data/' for refinement
    dataLoc = args.dataLocation

    print("Learning rate - {}.".format(lRate))
    print("Number of epochs - {}.".format(numEpoch))
    print("Adjust left and right angle by a factor of - {}".format(angleAdj))

    if baseModel:
        model = getModel(lRate)

        model = newModel(model, numEpoch, batch, angleAdj, dataLoc)
        saveModel(model, "./model_base.json", "./model_base.h5")
    else:
        print("Running in refine mode")
        with open("model_base.json", 'r') as file:
            model = model_from_json(json.load(file))

        model.compile(loss='mean_squared_error',
          optimizer=Adam(lr=lRate)
          )
        model.load_weights("model_base.h5")
        model = refineModel(model, numEpoch, batch, angleAdj, dataLoc)
        saveModel(model, "./{}.{}".format(saveModelFName, "json"), "./{}.{}".format(saveModelFName, "h5"))
