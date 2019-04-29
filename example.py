# the OFFICIAL MEMEFUL codebase for the 18.065 final project

# mnist ML tensorflow example

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def show_img(arr, text = ""):
    print(type(arr))
    if type(arr) != np.ndarray:
        raise Exception('The image could not be showed - not a numpy array')

    plt.figure(0)
    plt.imshow(arr, interpolation='nearest')
    plt.title(text)
    plt.show()


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


if __name__ == "__main__":
    # show one picture  of the train set
    show_img(x_train[0], "Expectation = "+ str(y_train[0]))