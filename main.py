# MAIN CODE FOR NEURAL MEMEFUL NETWORK

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import json
from meme_utils import *
import tensorflow as tf


def load_meme(id):
    # loads the image associated with the id of meme
    # returns an array of the rgb values of the image
    folder = '/Users/yoraish/Dropbox (MIT)/MIT/School/18.065/project/memes_resized'
    img = Image.open(os.path.join(folder,id+'.png'))
    return img




def create_model():
    # creates and trains a model to classify memes to one of the like categories

    # gather the data: x_train, y_train, x_test, y_test
    # x are lists of numpy arrays of images
    # y are lists of integers corresponding to the class of the image in the same index at x

    x_train = []
    y_train_ids = []
    y_train = []
    x_test = []
    y_test_ids = []
    y_test = []

    num_classes = 8

    # populate train ids
    with open("train_db.json") as train_db:
        # create a dict to map ids to class
        id_to_ups = json.load(train_db)
        y_train_ids = [id for id in id_to_ups.keys()]
        y_train = np.array([id_to_ups[id] for id in y_train_ids])

    # populate test ids
    with open("test_db.json") as test_db:
        # create a dict to map ids to class
        id_to_ups = json.load(test_db)
        y_test_ids = [id for id in id_to_ups.keys()]
        y_test = np.array([id_to_ups[id] for id in y_test_ids])

    # populate x train based on the order in y_train
    # creates a list of numpy arrays, in the order they appear in the y lists
    x_train = np.array([ np.asarray(load_meme(y_train_ids[i])) for i in range(len(y_train)) ])
    x_test = np.array([ np.asarray(load_meme(y_test_ids[i])) for i in range(len(y_test)) ])

    print(y_train)
    # define the model
    n = len(x_train[0]) # the size of the images
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(n,n,3)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    # [
    #     tf.keras.layers.Flatten(input_shape=(n,n ,3)), # n by n by RGB

    # tf.layers.conv2d(inputs=net, name='layer_conv1',
    #                        filters=32, kernel_size=3,
    #                        padding='same', activation=tf.nn.relu),

    # tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2),

    # tf.layers.conv2d(inputs=net, name='layer_conv2',
    #                        filters=64, kernel_size=3,
    #                        padding='same', activation=tf.nn.relu),

    # tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2),

    # tf.layers.conv2d(inputs=net, name='layer_conv3',
    #                        filters=64, kernel_size=3,
    #                        padding='same', activation=tf.nn.relu),

    # tf.layers.dense(inputs=net, name='layer_fc_2',
    #                     units=num_classes)
    # ])

    model.compile(optimizer='adam', 
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    print(len(x_train), len(y_train))
    model.fit(x_train, y_train, epochs=5)
    # test_loss, test_acc = model.evaluate(x_test[:1], y_test[:1])

if __name__ == '__main__':
    # create and train the model
    create_model()
