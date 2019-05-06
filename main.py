# MAIN CODE FOR NEURAL MEMEFUL NETWORK

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import json
from meme_utils import *

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

    # populate train ids
    with open("train_db.json") as train_db:
        # create a dict to map ids to class
        id_to_ups = json.load(train_db)
        y_train_ids = [id for id in id_to_ups.keys()]
        y_train = [id_to_ups[id] for id in y_train_ids]

    # populate test ids
    with open("test_db.json") as test_db:
        # create a dict to map ids to class
        id_to_ups = json.load(test_db)
        y_test_ids = [id for id in id_to_ups.keys()]
        y_test = [id_to_ups[id] for id in y_test_ids]

    # populate x train based on the order in y_train
    # creates a list of numpy arrays, in the order they appear in the y lists
    x_train = [ np.asarray(load_meme(y_train_ids[i])) for i in range(len(y_train)) ]
    x_test = [ np.asarray(load_meme(y_test_ids[i])) for i in range(len(y_test)) ]

    print(y_train[500], y_train_ids[500])



if __name__ == '__main__':
    # create and train the model
    create_model()
