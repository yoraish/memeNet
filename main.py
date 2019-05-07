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
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

def load_meme(id):
    # loads the image associated with the id of meme
    # returns an array of the rgb values of the image
    folder = '/Users/yoraish/Dropbox (MIT)/MIT/School/18.065/project/memes_resized'
    img = Image.open(os.path.join(folder,id+'.png'))
    # normalize image
    img = normalize(img)
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


def conv_net(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
    conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
    conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))

    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
    conv2_bn = tf.layers.batch_normalization(conv2_pool)
  
    # 5, 6
    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
    conv3_bn = tf.layers.batch_normalization(conv3_pool)
    
    # 7, 8
    conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv4_bn = tf.layers.batch_normalization(conv4_pool)
    
    # 9
    flat = tf.contrib.layers.flatten(conv4_bn)  

    # 10
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)
    
    # 11
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)
    
    # 12
    full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
    full3 = tf.nn.dropout(full3, keep_prob)
    full3 = tf.layers.batch_normalization(full3)    
    
    # 13
    full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
    full4 = tf.nn.dropout(full4, keep_prob)
    full4 = tf.layers.batch_normalization(full4)        
    
    # 14
    out = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=8, activation_fn=None)
    return out

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
    x_train = np.float32(np.array([ np.asarray(load_meme(y_train_ids[i])) for i in range(len(y_train)) ]))
    print("Generating Test Image Data")
    x_test = np.float32(np.array([ np.asarray(load_meme(y_test_ids[i])) for i in range(len(y_test)) ]))
    print('---Done preparing data---')

    '''
    print('sanity check, showing the fifth entry of train set')
    show_img(x_train[5])
    '''


    # define the model
    # global parameters needed for training:

    n = len(x_train[0]) # the size of the images
    input_shape = (n,n,3)

    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    #create model
    model = Sequential()
    #add model layers
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #--------------------------
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape)) 
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #--------------------------
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))






    #compile model using accuracy to measure model performance
    model.compile(loss= losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    #train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=65)




if __name__ == '__main__':
    # create and train the model
    create_model()
