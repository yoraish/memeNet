# test script
from meme_utils import *

# load an img
img = load_meme('7xw6ik', 64)
print(img[5][5])

#show it
show_img(img)

# convert to grayscale
gsc = img_to_gsc(img)

# show it
show_img(gsc)

