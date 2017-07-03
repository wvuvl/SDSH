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
from facehash_net import vgg_arg_scope
from facehash_net import vgg_19

HASH_SIZE = 24
BATCH_SIZE = 80

if __name__ == '__main__':
    sess = tf.Session()

    with open('items_train.pkl', 'rb') as pkl:
        items_train = pickle.load(pkl)
    with open('items_test.pkl', 'rb') as pkl:
        items_test = pickle.load(pkl)

    bp_train = facehashinput.BatchProvider(BATCH_SIZE, items_train, cycled=False)
    bp_test = facehashinput.BatchProvider(BATCH_SIZE, items_test, cycled=False)

    t_images, t_labels = bp_train.inputs()
    bp_test.t_images = t_images
    bp_test.t_labels = t_labels

    sess = tf.Session()
    model = vgg_f('imagenet-vgg-f.mat', t_images)

    outputs = model.net['fc8']

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    lc = tf.train.latest_checkpoint("F:\\tmp\\vgg\\")
    saver.restore(sess, lc)

    b = np.empty([0, HASH_SIZE])

    while True:
        feed_dict = bp_train.get_batch()
        if feed_dict is None:
            break

        result = sess.run(outputs, feed_dict)

        b = np.concatenate((b, result))

    b_dataset = np.copy(b)

    output = open('b_dataset.pkl', 'wb')
    pickle.dump(b_dataset, output)
    output.close()

    b = np.empty([0, HASH_SIZE])

    while True:
        feed_dict = bp_test.get_batch()
        if feed_dict is None:
            break

        result = sess.run(outputs, feed_dict)

        b = np.concatenate((b, result))

    b_test = np.copy(b)

    output = open('b_test.pkl', 'wb')
    pickle.dump(b_test, output)
    output.close()
