import os
import lmdb
import numpy as np
from random import shuffle
import random
import pickle

images = []

path = "../data/nus_wide"

for root, dirs, files in os.walk(os.path.join(path, "image")):
    root_ = os.path.basename(root)
    for f in files:
        key = "{0}\{1}".format(root_, f)
        images.append(key)
        
images = set(images)
        
with open(os.path.join(path, "ImageList/ImageList.txt")) as f:
    content = f.readlines()
content = [x.strip() for x in content]

print(len(content))
labels = {}

for root, dirs, files in os.walk(os.path.join(path, "Groundtruth/AllLabels")):
    for f in files:
        print(f)
        with open(os.path.join(os.path.join(path, "Groundtruth/AllLabels"), f)) as file:
            lcontent = file.readlines()
        attribute = np.asarray([int(x.strip()) for x in lcontent])
        labels[f] = attribute
        
labels = labels.items()

labels = sorted(labels, key=lambda l: np.sum(l[1]))

for (l, a) in labels:
    print("{0} {1}".format(l, np.sum(a))) 
    


for (l, a) in labels:
    print("{0} {1}".format(l, np.sum(a))) 

labels = [(l[7:-4], a) for (l, a) in labels]

for (l, a) in labels:
    print("{0} {1}".format(l, np.sum(a))) 

keptlabels = labels[-21:]


labels_ids = {}
with open('labels.txt', 'w') as f:
    i = 0
    for (l, a) in labels:
        id = 1 << i
        print("{0} {1}".format(id, l))
        i += 1
        labels_ids[l] = id

print(labels_ids)

items = {}
number_of_two_and_more = 0

categorized = {key: [] for key in range(81)}

bitfilter = 0
for (l,a) in keptlabels:
    bitfilter |= labels_ids[l]

#import ipdb; ipdb.set_trace()
print('kept labels {}'.format(keptlabels))

for i in range(len(content)):
    label = 0
    k = 0
    for (l, a) in labels:
        if a[i] == 1:
            label |= labels_ids[l]
            k += 1
    if k > 1:
        number_of_two_and_more += 1
    if label != 0 and content[i] in images and (label & bitfilter) != 0:
        items[i] = ((label, content[i]))
        #print((label, content[i]))
        for id in range(81):
            if (label & (1<<id)) != 0:
                categorized[id].append(i)


print("Count of items with at least one label: {0}".format(len(items)))
print("Count of items with more than one label: {0}".format(number_of_two_and_more))




test_data = []

items_train = []
items_test = []

while len(items_test) < 5000:
    ind = random.randint(0,len(content))
    if ind in items:
        items_test.append(items[ind])
        test_data.append(ind )



test_data = set(test_data)



while len(items_train) < 10000:
    ind = random.randint(0,len(content))
    if ind not in test_data and ind in items:
        items_train.append(items[ind])


items_database = []

for i in range(len(content)):
    if i not in test_data and i in items:
        items_database.append(items[i])



print("Count of train items: {0}".format(len(items_train)))


shuffle(items_train)
shuffle(items_test)
shuffle(items_database)


if not os.path.exists('temp'):   
    os.makedirs('temp')   

output = open('../temp/items_uniform_train_nuswide.pkl', 'wb')   
pickle.dump(items_train, output
output.close()   

output = open('../temp/items_uniform_test_nuswide.pkl', 'wb')   
pickle.dump(items_test, output)
output.close()

output = open('../temp/items_uniform_db_nuswide.pkl', 'wb')   
pickle.dump(items_database, output)
output.close()

def write_txt_file(data,name):
    with open('../temp/{}.txt'.format(name),'w') as outF:
        for (label,fname) in data:
            fname = fname.replace('\\','/')
            outF.write('{}/out/{} '.format(os.getcwd(),fname))
            for i in range(81):
                if label & (1<<i) == 0:
                    outF.write('0 ')
                else:
                    outF.write('1 ')
            outF.write('\n')

write_txt_file(items_database,'database')
write_txt_file(items_test,'test')
write_txt_file(items_train,'train')



