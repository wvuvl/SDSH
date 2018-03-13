import os
import numpy as np
from random import shuffle
import random
import pickle
import matplotlib.pyplot as plt
from scipy import misc

images = []

path = "../data/imagenet"

with open(os.path.join(path,"imagenet.pkl"), 'rb')    as pkl:
    content = pickle.load(pkl)

validation_images = []

for root, dirs, files in os.walk("F:/DeepLearningCode/ImageNet/ILSVRC2012_img_val"):
    for f in files:
        if f[-5:] == ".JPEG":
            validation_images.append(f)

for root, _, files in os.walk(os.path.join(path)):
    root_ = os.path.basename(root)
    for f in files:
        if f[-5:] == ".JPEG":
            key = f
            images.append(key)
        
images = set(images)

with open(os.path.join(path, "ILSVRC2012_validation_ground_truth.txt")) as f:
    val = f.readlines()
val_labels = [int(x.strip()) for x in val]


print(len(content))

test_data = set()
train_data = set()

items_train = []
items_test = []
items_database = []

valid_labels = []

filenames = {}
ifilenames = {}
labels = {}

for id, label, filename in content:
    print(label)
    filenames[id] = filename
    labels[id] = label
    ifilenames[filename] = id
    valid_labels.append(id)

for i in range(len(validation_images)):
    v = val_labels[i]
    f = validation_images[i]
    if v in valid_labels:
        items_test.append((v, f))

categorized = {val: [] for (key, val) in filenames.items()}

for im in images:
    f = im[:9]
    l = ifilenames[f]
    categorized[f].append((l, im))

for (f, c) in categorized.items():
    shuffle(c)
    items_train += c[:100]
    items_database += c

print("Count of test items: {0}".format(len(items_test)))
print("Count of train items: {0}".format(len(items_train)))
print("Count of database items: {0}".format(len(items_database)))

shuffle(items_train)
shuffle(items_test)
shuffle(items_database)

#
# for (l, f) in items_train:
#     im = misc.imread("../data/imagenet/" + f)
#     print(labels[l])
#     plt.imshow(im, interpolation='nearest')
#     plt.show()

if not os.path.exists('../temp'):
    os.makedirs('../temp')

output = open('../data/imagenet/items_train_imagenet.pkl', 'wb')
pickle.dump(items_train, output, protocol=1)
output.close()

output = open('../data/imagenet/items_test_imagenet.pkl', 'wb')
pickle.dump(items_test, output, protocol=1)
output.close()

output = open('../data/imagenet/items_db_imagenet.pkl', 'wb')
pickle.dump(items_database, output, protocol=1)
output.close()
