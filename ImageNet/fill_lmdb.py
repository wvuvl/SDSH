import os
import lmdb
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image
import sys
try:
	from BytesIO import BytesIO
except ImportError:
	from io import BytesIO

path = "../data/imagenet"

def run(file):
	image = misc.imread(file)

	w = image.shape[1]
	h = image.shape[0]

	size = 224

	if w > h:
		x_margin_left = (w - h) // 2
		x_margin_right = w - h - x_margin_left
		image = image[:, x_margin_left:-x_margin_right]


	if w < h:
		y_margin_up = (h - w) // 2
		y_margin_down = h - w - y_margin_up
		image = image[y_margin_up:-y_margin_down, :]

	image = misc.imresize(image, (224, 224), interp='bilinear')

	return image

#img = np.fromstring(, dtype=np.uint8)
#img = np.reshape(img, (3, 32, 32))
#img = np.transpose(img, (1, 2, 0))

#run("image/chapel/0371_208781723.jpg")
#run("image/chapel/0366_415591917.jpg")

def saveAllToDB(env):
	k = 0
	with env.begin(write=True) as txn:
		for root, dirs, files in os.walk(os.path.join(path)):
			root_ = os.path.basename(root)
			for f in files:
				key = f
				buf = txn.get(key.encode('ascii'))
				if f[-5:] == ".JPEG" and buf is None:
					try:
						print(key)
						im = run(os.path.join(root, f))
						buffer = BytesIO()
						im = Image.fromarray(im)
						im.save(buffer, format="jpeg")
						buffer.seek(0)
						txn.put(key.encode('ascii'), buffer.read())
						k += 1
					except:
						the_type, the_value, the_traceback = sys.exc_info()
						print(the_value)
						print(the_traceback)

						print("FAILED")
	return k


env = lmdb.open(os.path.join(path,"imagenet"), map_size= 8 * 1024 * 1024 * 1024)
k = saveAllToDB(env)

print("Done {0}".format(k))
