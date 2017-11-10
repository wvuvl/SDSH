import os
import lmdb
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image
try:
	from BytesIO import BytesIO
except ImportError:
	from io import BytesIO


def extractAllFromDB(env):
	k = 0
	with env.begin(write=True) as txn:
		for root, dirs, files in os.walk("image"):
			root_ = os.path.basename(root)
			for f in files:
				try:
					key = "{0}\{1}".format(root_, f)
					print(key)
					buf = txn.get(key.encode('ascii'))
					location = "out/" + key
					directory = os.path.dirname(location)
					print(directory)
					if not os.path.exists(directory):
						os.makedirs(directory)
					with open(location,'wb') as out:
						out.write(buf)
					k += 1
				except:
					print("FAILED")
	return k

env = lmdb.open('nuswide', map_size= 8 * 1024 * 1024 * 1024)
k = extractAllFromDB(env)

print("Done {}", k)

#with env.begin(write=True) as txn:
#	for root, dirs, files in os.walk("image"):
#		root_ = os.path.basename(root)
#		for f in files:
#			key = "{0}\{1}".format(root_, f)
#			print(key)
#			buffer = BytesIO()
#			buf = txn.get(key.encode('ascii'))
#			buffer.write(buf)
#			image = misc.imread(buffer)
#			plt.imshow(image)
#			plt.show()
