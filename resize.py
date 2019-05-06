# resize numpy array (image) to a set dimension
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import scipy.misc
import os

verify = False # if True then will load images after saving 

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
            # generate the path to the image
            img_path = os.path.join(path, filename)

            #load the image - with scipy to get lists of lists of three items (RGB)
            img = scipy.misc.imread(img_path, flatten=False, mode='RGB')
            # resize the loaded image
            resized_img = resize(img, c,r)

            #save the image
            resized_img_path = os.path.join(save_to_folder, filename[:-3] + 'png')
            cv2.imwrite(resized_img_path, resized_img)


            if verify:
                # loadad an image and see how it is constructed
                print('IN VERIFY MODE')
                img_open = scipy.misc.imread(resized_img_path, flatten=False, mode='RGB')
                show_img(img_open)
                print(img_open)

            
        else:
            continue

if __name__ == "__main__":

    resize_all_in_folder("/Users/yoraish/Dropbox (MIT)/MIT/School/18.065/project/memes","/Users/yoraish/Dropbox (MIT)/MIT/School/18.065/project/memes_resized", 128,128) # change the path to be true to your machine
