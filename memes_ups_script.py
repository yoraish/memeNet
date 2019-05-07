# code to create a json object that maps meme id's to their upvote-class
# max ups is 293544
# min ups is 494
# on a log base 2 scale those values will be in the inclusive range of [9,16]
# subtracting  9 to bring to range [0,7] only two entries are at value 2 - put them in 7 too


# THIS SCRIPT WILL ONLY WORK IF THE DIRECTORY MEMES_RESIZED EXISTS!!
# VERIFIES THE EXISTENCE OF FILES WHEN ADDING THEIR INFO TO THE DATABASE

import json
from math import log
import matplotlib.pyplot as plt
import numpy as np
import os




def vis(dict, heading):
    plt.figure(0)
    lst = sorted([i for i in dict.values()])

    plotting_bins=np.arange(0,10,1) # need to add an extra bin when plotting 
    plt.hist(lst, bins=plotting_bins)
    plt.title(heading)
    plt.show()


memes_dict = {}
with open("../db.json") as db:
    data = json.load(db)
    type(data)
    for i in range(1,len(data['_default'])):
        id = data['_default'][str(i)]['id']
        ups = data['_default'][str(i)]['ups']

        if ups > 2**16 -1:
            ups = 2**16
        
        print('---->',data['_default'][str(i)]['id'], data['_default'][str(i)]['ups'], '--->', round(log(ups,2))-9)

        # check that a file with this id exists in the resized directory, 
        # only store info if this file exists
        if os.path.isfile('../memes_resized/'+id + '.png'):
            memes_dict[id] = round(log(ups,2))-9

    # split the data to 75% train and 25% test
    train_data = {}
    test_data = {}

    # sort the data to split it in a meaningful way
    lst = sorted([(value, id) for id,value in memes_dict.items()])

    counter = 0
    for tup in lst:
        # if multiple of 4 - put in test
        # else put in train
        id = tup[1]
        value = tup[0]
        if counter%9 == 0:
            test_data[id] = value
        else:
            train_data[id] = value
        counter +=1

    vis(test_data, 'Histogram for test data')
    vis(train_data, 'Histogrtam for train data')

    with open('memes_ups_db.json', 'w') as json_file:  
        json.dump(memes_dict, json_file)

    # save the dict as a json file
    with open('train_db.json', 'w') as json_file:  
        json.dump(train_data, json_file)

    with open('test_db.json', 'w') as json_file:  
        json.dump(test_data, json_file)
