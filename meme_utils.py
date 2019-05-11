# helper functions to help the help

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import json

def show_img(arr, text = ""):
    if type(arr) != np.ndarray:
        raise Exception('The image could not be showed - not a numpy array')

    plt.figure(0)
    plt.imshow(arr, interpolation='nearest')
    plt.title(text)
    plt.show()


def normalize(x):
    '''
    takes in a numpy array of [r,c,3]
    returns normalized x
    '''
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val)/(max_val-min_val)
    return x

def load_meme(id,size):
    # loads the image associated with the id of meme
    # returns an array of the rgb values of the image
    folder = '/Users/yoraish/Dropbox (MIT)/MIT/School/18.065/project/memes_resized_'+str(size)
    img = Image.open(os.path.join(folder,id+'.png'))
    # normalize image
    img = normalize(img) # maybe take out - check both
    img = reduce_color_precision(img)
    return img

def img_to_gsc(x):
    # takes in a numpy array of an image (n,n,3)
    # returns grayscale x of size (n,n,3)
    new = np.zeros( x.shape)
    for i in range(len(x)):
        for j in range(len(x[0])):
            gsc = round(0.3*x[i][j][0] + 0.59*x[i][j][1] + 0.11*x[i][j][2],1)
            new[i][j] = np.array([gsc,gsc,gsc])

    return new

def reduce_color_precision(x):
    # take in a numpy array of an image
    # returns a reduced accuracy array to 0.x

    for i in range(len(x)):
        for j in range(len(x[0])):
            for k in range(len(x[0][0])):
                x[i][j][k] = round(x[i][j][k],1)
    return x