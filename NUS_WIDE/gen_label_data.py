import os
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
        
with open(os.path.join(path, "ImageList/Imagelist.txt")) as f:
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
    
labels = [(l[7:-4], a) for (l, a) in labels]

for (l, a) in labels:
    print("{0} {1}".format(l, np.sum(a))) 

keptlabels = labels[-21:]

print('kept')

for (l, a) in keptlabels:
    print("{0} {1}".format(l, np.sum(a))) 
#labels = labels[-21:]

labels_ids = {}
with open('labels.txt', 'w') as f:
    i = 0
    for (l, a) in reversed(labels):
        id = 1 << i
        print("{0} {1}".format(id, l))
        i += 1
        labels_ids[l] = id

print(labels_ids)

items = {}
number_of_two_and_more = 0

categorized = {key: [] for (key, _) in labels}

bitfilter = 0
for (l, a) in keptlabels:
    bitfilter |= labels_ids[l]

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
        for l, id in labels_ids.items():
            if (label & id) != 0:
                categorized[l].append(i)

keptlabels_ = [l for (l, a) in keptlabels]

categorized = {k: v for (k, v) in categorized.items() if k in keptlabels_}

print("Count of items with at least one label: {0}".format(len(items)))
print("Count of items with more than one label: {0}".format(number_of_two_and_more))

styles= {
    '2100.10500', # 100 per label out of 21 for test, 500 per label out of 21 for train, train and rest for db (db contains train)
    '2100._', # 1000 per label out of 21 for test, rest for train and db (db equals to train)
    '5000.10000', # 5000 randomly sampled for test and 10000 randomly sampled for train, train and rest for db (db contains train)
}

def generate(style):
    print("Running:")
    print(style)

    test_data = set()
    train_data = set()
    
    items_train = []
    items_test = []
    items_database = []

    if style == '2100._':
        for label ,l in categorized.items():
            shuffle(l)
            num = 0
            for l_it in l:
                if l_it not in test_data and l_it in items:
                    test_data.add(l_it)
                    items_test.append(items[l_it])
                    num += 1
                    if num == 100:
                        break
    
        for i in range(len(content)):
            if i not in test_data and i in items:
                items_train.append(items[i])
    
    elif style == '2100.10500':
        for label ,l in categorized.items():
            shuffle(l)
            num = 0
            for l_it in l:
                if l_it not in test_data and l_it in items:
                    test_data.add(l_it)
                    items_test.append(items[l_it])
                    num += 1
                    if num == 100:
                        break
            shuffle(l)
            num = 0
            for l_it in l:
                if (l_it not in test_data) and (l_it not in train_data) and (l_it in items):
                    train_data.add(l_it)
                    items_train.append(items[l_it])
                    num += 1
                    if num == 500:
                        break
    
        for i in range(len(content)):
            if i not in test_data and i in items:
                items_database.append(items[i])
    
    elif style == '5000.10000':
        while len(items_test) < 5000:
            ind = random.randint(0, len(content))
            if ind not in test_data and ind in items:
                items_test.append(items[ind])
                test_data.add(ind)
        
        while len(items_train) < 10000:
            ind = random.randint(0, len(content))
            if (ind not in test_data) and (ind not in train_data) and (ind in items):
                items_train.append(items[ind])
                train_data.add(ind)
    
        for i in range(len(content)):
            if i not in test_data and i in items:
                items_database.append(items[i])
    
    print("Count of test items: {0}".format(len(items_test)))
    print("Count of train items: {0}".format(len(items_train)))
    print("Count of database items: {0}".format(len(items_database)))
    
    shuffle(items_train)
    shuffle(items_test)
    shuffle(items_database)
    
    if not os.path.exists('../temp'):
        os.makedirs('../temp')
    
    output = open('../temp/items_train_nuswide_{}.pkl'.format(style), 'wb')
    pickle.dump(items_train, output)
    output.close()
    
    output = open('../temp/items_test_nuswide_{}.pkl'.format(style), 'wb')
    pickle.dump(items_test, output)
    output.close()
    
    output = open('../temp/items_db_nuswide_{}.pkl'.format(style), 'wb')
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
    
    if style == '5000.10000':
        write_txt_file(items_database,'database')
        write_txt_file(items_test,'test')
        write_txt_file(items_train,'train')

for s in styles:
    generate(s)
