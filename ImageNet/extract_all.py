import os
import shutil
from random import shuffle
import random
import pickle
import tarfile

path = "../data/imagenet"

with open(os.path.join(path,"imagenet.pkl"), 'rb')    as pkl:
	content = pickle.load(pkl)

for id, label, filename in content:
	print(id, label, filename)
	tar_path = os.path.join(path, "{}.tar".format(filename))
	
	tar = tarfile.open(tar_path)
	tar.extractall(path)
	tar.close()
