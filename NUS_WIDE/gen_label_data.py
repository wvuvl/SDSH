import os
import lmdb
import numpy as np
from random import shuffle
import pickle

images = []

for root, dirs, files in os.walk("image"):
	root_ = os.path.basename(root)
	for f in files:
		key = "{0}\{1}".format(root_, f)
		images.append(key)
		
images = set(images)
		
with open("ImageList/ImageList.txt") as f:
    content = f.readlines()
content = [x.strip() for x in content]

print(len(content))
labels = {}

for root, dirs, files in os.walk("Groundtruth/AllLabels"):
	for f in files:
		print(f)
		with open(os.path.join("Groundtruth/AllLabels", f)) as file:
			lcontent = file.readlines()
		attribute = np.asarray([int(x.strip()) for x in lcontent])
		labels[f] = attribute
		
labels = labels.items()

labels = sorted(labels, key=lambda l: np.sum(l[1]))

for (l, a) in labels:
	print("{0} {1}".format(l, np.sum(a))) 
	
labels = labels[-21:]

for (l, a) in labels:
	print("{0} {1}".format(l, np.sum(a))) 

labels = [(l[7:-4], a) for (l, a) in labels]

for (l, a) in labels:
	print("{0} {1}".format(l, np.sum(a))) 

	
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

categorized = {key: [] for key in range(21)}

for i in range(len(content)):
	label = 0
	k = 0
	for (l, a) in labels:
		if a[i] == 1:
			label |= labels_ids[l]
			k += 1
	if k > 1:
		number_of_two_and_more += 1
	if label != 0 and content[i] in images:
		items[i] = ((label, content[i]))
		#print((label, content[i]))
		for id in range(21):
			if (label & (1<<id)) != 0:
				categorized[id].append(i)
		
print("Count of items with at least one label: {0}".format(len(items)))
print("Count of items with more than one label: {0}".format(number_of_two_and_more))


test_data = []

items_train = []
items_test = []

for id in range(21):
	l = categorized[id]
	shuffle(l)
	l = l[:100]
	test_data += l
	items_test += [items[i] for i in l]
	
print("Count of test items: {0}".format(len(items_test)))

test_data = set(test_data)

for i in range(len(content)):
	if i not in test_data and i in items:
		items_train.append(items[i])
		
print("Count of train items: {0}".format(len(items_train)))


shuffle(items_train)
shuffle(items_test)   

if not os.path.exists('temp'):   
    os.makedirs('temp')   

output = open('temp/items_train_nuswide.pkl', 'wb')   
pickle.dump(items_train, output)   
output.close()   

output = open('temp/items_test_nuswide.pkl', 'wb')   
pickle.dump(items_test, output)   
output.close()

