# resize numpy array (image) to a set dimension
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def show_img(arr, text = ""):
    if type(arr) != np.ndarray:
        raise Exception('The image could not be showed - not a numpy array')

    plt.figure(0)
    plt.imshow(arr, interpolation='nearest')
    plt.title(text)
    plt.show()

def resize(orig, c,r):
    show_img(orig)
    show_img(cv2.resize(orig,  dsize=(r, c), interpolation=cv2.INTER_CUBIC))

    


if __name__ == "__main__":
    arr = np.asarray(Image.open('sample_meme.png'))
    resize(arr,50,50)