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
    return cv2.resize(orig,  dsize=(r, c), interpolation=cv2.INTER_CUBIC)

    
def resize_all_in_folder(path, save_to_folder, r, c):
    # take all the files in a given directory and save them with the same name, to a new directory - resized to a uniform size
    for filename in os.listdir(path):
        if filename.endswith(".png") or filename.endswith(".jpg"): 
            print('resizing ', filename)
            img_path = os.path.join(path, filename)

            #load the image
            img = np.asarray(Image.open(img_path))
            resized_img = resize(img, c,r)
            #save the image
            cv2.imwrite(os.path.join(save_to_folder, filename[:-3] + 'png'), resized_img)

            
        else:
            continue

if __name__ == "__main__":
    # arr = np.asarray(Image.open('sample_meme_2.jpg'))
    # resize(arr,244,244)

    resize_all_in_folder("/Users/yoraish/Dropbox (MIT)/MIT/School/18.065/project/memes","/Users/yoraish/Dropbox (MIT)/MIT/School/18.065/project/memes_resized", 244,244) # change the path to be true to your machine