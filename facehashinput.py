#! python3
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from contextlib import closing
from scipy import misc
from random import shuffle
import mmap
import random


class Reader:
    def __init__(self, path, items=None, train=True, test=False):
        self.path = path
        
        self.items = []
        
        height = 32
        width = 32
        depth = 3
        
        self.label_bytes = 1
        self.image_bytes = height * width * depth
        self.record_bytes = self.label_bytes + self.image_bytes

        if items is not None:
            self.items = items
        else:
            if train:
                self.read_batch('data_batch_1.bin')
                self.read_batch('data_batch_2.bin')
                self.read_batch('data_batch_3.bin')
                self.read_batch('data_batch_4.bin')
                self.read_batch('data_batch_5.bin')

            if test:
                self.read_batch('test_batch.bin')

    def read_batch(self, batch):
        with open(os.path.join(self.path, batch), 'rb') as f:
            with closing(mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)) as m:
                for i in range(10000):
                    label = m[i * self.record_bytes]
                    img = np.fromstring(
                        m[i * self.record_bytes + self.label_bytes:i * self.record_bytes + self.record_bytes],
                        dtype=np.uint8)
                    img = np.reshape(img, (3, 32, 32))
                    img = np.transpose(img, (1, 2, 0))
#                   plt.imshow(img, interpolation='nearest')
#                   plt.show()
                    self.items.append((label, img))

    def get_labels(self):
        return [item[0] for item in self.items]

    def get_images(self):
        return [item[1] for item in self.items]


class BatchProvider:
    def __init__(self, batch_size, items, cycled=True):
        self.items = items
        self.batch_size = batch_size
        depth = 3
        self.t_images = tf.placeholder(tf.float32, [None, 224, 224, depth])
        self.t_labels = tf.placeholder(tf.int32, [None, 1])

        self.current_image = 0
        self.cycled = cycled
        self.done = False

    def inputs(self):
        return self.t_images, self.t_labels

    def get_batch(self):
        b_images = []
        b_labels = []

        for i in range(0, self.batch_size):
            if self.current_image == len(self.items):
                if self.cycled:
                    self.current_image = 0
                    shuffle(self.items)
                else:
                    return None
            im = misc.imresize(self.items[self.current_image][1], (224, 224), interp='bilinear')

            #if random.random() > 0.5:
            #    im = np.fliplr(im)
            #im = np.roll(im, random.randint(-21, 21), 0)
            #im = np.roll(im, random.randint(-21, 21), 1)
            #plt.imshow(im)
            #plt.show()

            b_images.append(im)
            b_labels.append([self.items[self.current_image][0]])

            self.current_image += 1

        feed_dict = {self.t_images: b_images, self.t_labels: b_labels}

        return feed_dict

if __name__ == '__main__':
    r = Reader('cifar-10-batches-bin', True)

    p = BatchProvider(5, r.items)

    b = p.get_batch()

    ims = b[p.t_images]
    for im in ims:
        plt.imshow(im, interpolation='nearest')
        plt.show()
