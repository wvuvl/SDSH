#! python3
import facehashinput
import pickle
from random import shuffle

r = facehashinput.Reader('cifar-10-batches-bin', None, True, True)


categorized = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

for item in r.items:
    categorized[item[0]].append(item[1])

items_train = []
items_test = []

for i in range(10):
    shuffle(categorized[i])
    items_test += [(i, x) for x in categorized[i][0:1000]]
    items_train += [(i, x) for x in categorized[i][1000:]]

shuffle(items_train)
shuffle(items_test)

output = open('items_train.pkl', 'wb')
pickle.dump(items_train, output)
output.close()

output = open('items_test.pkl', 'wb')
pickle.dump(items_test, output)
output.close()
