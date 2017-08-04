#! python3
import numpy as np
import facehashinput
import pickle
from compute_S import compute_s
from return_map import return_map
from datetime import datetime
from facehash_net import vgg_f
import time
import tensorflow as tf

HASH_SIZE = 24
BATCH_SIZE = 80

def GenHashes(t_images, t_labels, outputs, sess, items_train, items_test):
    bp_train = facehashinput.BatchProvider(BATCH_SIZE, items_train, cycled=False)
    bp_test = facehashinput.BatchProvider(BATCH_SIZE, items_test, cycled=False)

    b = np.empty([0, HASH_SIZE])

    while True:
        feed_dict = bp_train.get_batch()
        if feed_dict is None:
            break

        result = sess.run(outputs, {t_images: feed_dict["images"], t_labels: feed_dict["labels"]})

        b = np.concatenate((b, result))

    b_dataset = np.copy(b)


    b = np.empty([0, HASH_SIZE])

    while True:
        feed_dict = bp_test.get_batch()
        if feed_dict is None:
            break

        result = sess.run(outputs, {t_images: feed_dict["images"], t_labels: feed_dict["labels"]})

        b = np.concatenate((b, result))

    b_test = np.copy(b)

    return b_dataset, b_test

if __name__ == '__main__':
    sess = tf.Session()

    t_images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    t_labels = tf.placeholder(tf.int32, [None, 1])

    model = vgg_f('imagenet-vgg-f.mat', t_images)
    outputs = model.net['fc8']

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    lc = tf.train.latest_checkpoint("F:\\tmp\\vgg\\")
    saver.restore(sess, lc)

    with open('items_train.pkl', 'rb') as pkl:
        items_train = pickle.load(pkl)
    with open('items_test.pkl', 'rb') as pkl:
        items_test = pickle.load(pkl)

    b_dataset, b_test = GenHashes(t_images, t_labels, outputs, sess, items_train, items_test)

    output = open('b_dataset.pkl', 'wb')
    pickle.dump(b_dataset, output)
    output.close()
    output = open('b_test.pkl', 'wb')
    pickle.dump(b_test, output)
    output.close()
