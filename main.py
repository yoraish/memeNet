# MAIN CODE FOR NEURAL MEMEFUL NETWORK

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import json
from meme_utils import *

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_meme(id,size):
    # loads the image associated with the id of meme
    # returns an array of the rgb values of the image
    folder = '/Users/yoraish/Dropbox (MIT)/MIT/School/18.065/project/memes_resized_'+str(size)
    img = Image.open(os.path.join(folder,id+'.png'))
    # normalize image
    img = normalize(img) # maybe take out - check both
    return img

def encode_labels(labels_list):
    """takes in a labels list and returns a 2D array mapping id (int) to label
    
    Arguments:
        labels_list {list} -- the list of NAMES of labels (in our memes case would be ints as well)
    """
    encoded = np.zeros((len(labels_list), 8 )) # 8 for 8 classes of labels
    for ix, label in enumerate(labels_list):
        encoded[ix][label] = 1
    return encoded

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate

def create_model(size = 28, get_meme_data = True):
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

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # populate train ids 
    if get_meme_data:
        num_outputs = 8

        print("Generating Training Label Data")
        with open("train_db.json") as train_db:
            # create a dict to map ids to class
            id_to_ups = json.load(train_db)
            y_train_ids = [id for id in id_to_ups.keys()]
            y_train = np.array([id_to_ups[id] for id in y_train_ids])

        # populate test ids
        print("Generating Test Label Data")
        with open("test_db.json") as test_db:
            # create a dict to map ids to class
            id_to_ups = json.load(test_db)
            y_test_ids = [id for id in id_to_ups.keys()]
            y_test = np.array([id_to_ups[id] for id in y_test_ids])

        # populate x train based on the order in y_train
        # creates a list of numpy arrays, in the order they appear in the y lists
        print("Generating Training Image Data")
        x_train = np.float32(np.array([ np.asarray(load_meme(y_train_ids[i], size)) for i in range(len(y_train)) ]))
        print("Generating Test Image Data")
        x_test = np.float32(np.array([ np.asarray(load_meme(y_test_ids[i],size)) for i in range(len(y_test)) ]))
        print('---Done preparing data---')
    
    else:
        size = 32
        num_outputs = 10

        from tensorflow.keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # print('sanity check, showing the fifth entry of train set')
    # show_img(x_train[5])
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++



    # define the model
    # global parameters needed for training:

    input_shape = (size,size,3)

    #one-hot encode target column
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #create model 
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    #z-score
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)


    y_train = to_categorical(y_train,num_outputs)
    y_test = to_categorical(y_test,num_outputs)
    
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(num_outputs, activation='softmax'))
    
    model.summary()
    
    #data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        )
    datagen.fit(x_train)
    
    #training
    batch_size = 64
    
    opt_rms = tf.keras.optimizers.RMSprop(lr=0.001,decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                        steps_per_epoch=x_train.shape[0] // batch_size,epochs=125,\
                        verbose=1,validation_data=(x_test,y_test),callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule)])
    #save to disk
    # model_json = model.to_json()
    # with open('model.json', 'w') as json_file:
    #     json_file.write(model_json)
    # model.save_weights('model.h5') 
    
    #testing
    scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))



    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++





if __name__ == '__main__':
    # create and train the model
    # choose image size from 28,64,128,244
    # get meme data in the specified image size, or not, and then the data will be CIFAR10
    create_model(32, get_meme_data = True)
