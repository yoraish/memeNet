# helper functions to help the help

import numpy as np
import matplotlib.pyplot as plt

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