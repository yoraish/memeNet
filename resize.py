# resize numpy array (image) to a set dimension
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import os



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

    
def resize_all_in_folder(path, save_to_folder):
    for filename in os.listdir(path):
        if filename.endswith(".png") or filename.endswith(".jpg"): 
            print(os.path.join(path, filename))
            continue
        else:
            continue

if __name__ == "__main__":
    # arr = np.asarray(Image.open('sample_meme_2.jpg'))
    # resize(arr,244,244)

    resize_all_in_folder("/Users/yoraish/Dropbox (MIT)/MIT/School/18.065/project/memes","") # change the path to be true to your machine