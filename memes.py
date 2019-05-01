# sample code to play around with the dataset
import json
memes_dict = {}
with open("../db.json") as db:
    data = json.load(db)
    type(data)
    for i in range(1,len(data['_default'])):
        id = data['_default'][str(i)]['id']
        ups = data['_default'][str(i)]['ups']
        
        print('---->',data['_default'][str(i)]['id'], data['_default'][str(i)]['ups'])
        memes_dict[id] = ups
    print(memes_dict)
    # save the dict as a json file
    with open('memes_ups_db.json', 'w') as json_file:  
        json.dump(memes_dict, json_file)