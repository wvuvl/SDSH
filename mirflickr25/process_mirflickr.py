import pickle
from functools import reduce
import shutil
import glob
import random
#import ipdb
def num_to_key(num):
    return './mirflickr/im{}.jpg'.format(num)

img_files = glob.glob('./mirflickr/im*.jpg')

meta_db = {f:[] for f in img_files}

labelFiles = glob.glob('./*.txt')
labelFiles = sorted(labelFiles)
for i,labelFile in enumerate(labelFiles):
    with open(labelFile) as inF:
        images = inF.readlines()
        images = [i.strip() for i in images]
    for image in images:
        meta_db[num_to_key(image)].append(1<<i)


imgs = img_files
random.shuffle(imgs)
testing = imgs[:2000]
training = imgs[2000:]
db = training

def make_pkl(samples,fname):
    data = []
    for sample in samples:

        label = 0
        for l in meta_db[sample]:
            label |= l

        data.append((label,sample))
        
    with open(fname,'wb') as outF:
        pickle.dump(data,outF)
    del data

make_pkl(testing,'mirflickr25test.pkl')
#ipdb.set_trace()
make_pkl(training,'mirflickr25train.pkl')
shutil.copyfile('mirflickr25train.pkl','mirflickr25db.pkl') 
#make_pkl(database,'mirflickr25db.pkl')




