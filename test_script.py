# test script
from meme_utils import *
import json
from math import log

"""
# load an img
img = load_meme('7xw6ik', 64)
print(img[5][5])

#show it
show_img(img)

# convert to grayscale
gsc = img_to_gsc(img)

# show it
show_img(gsc)

"""


# try and create a dictionary of all the words in the titles
words = {} #" maps words to the number of times seen"
title_to_ups = {}

srt_lst = []
with open("../db.json") as db:
    data = json.load(db)
    for i in range(1,len(data['_default'])):
        title = data['_default'][str(i)]['title'].lower()
        ups = data['_default'][str(i)]['ups']
        title_to_ups[title] = ups # <----------------maybe log scale here

        # find the words in the title
        wlst = title.split()
        # put in words dict
        for w in wlst:
            if w in words.keys():
                words[w]+= 1
            else:
                words[w] = 1
print(words)


