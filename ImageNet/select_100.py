import os
import shutil
from random import shuffle
import random
import pickle

path = "../data/imagenet"

with open(os.path.join(path, "labels.txt")) as f:
    content = f.readlines()
content = [x.strip().split(';') for x in content]
content = [(int(x[0]), x[1], x[2]) for x in content]

shuffle(content)

content = content[:100]

src_path = 'F:/DeepLearningCode/ImageNet/ILSVRC2012_img_train'

for id, label, filename in content:
	print(id, label, filename)
	shutil.copy(os.path.join(src_path, "{}.tar".format(filename)), os.path.join(path, "{}.tar".format(filename)))
	
output = open(os.path.join(path,"imagenet.pkl"), 'wb')   
pickle.dump(content, output)
output.close()
