#! python3
import logging
import os
import pickle
import sys
import time
from pprint import pformat

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import batch_provider
import constructor
import numpy as np


class PCA:
    def run(self):
        with tf.Graph().as_default(), tf.Session() as session:
            with open('temp/items_train.pkl', 'rb') as pkl:
                items_train = pickle.load(pkl)

            BATCH_SIZE = 150

            bp = batch_provider.BatchProvider(BATCH_SIZE, items_train, cycled=False)

            model = constructor.net(BATCH_SIZE, 24, 0, 0)

            output = model.net['relu7']

            session.run(tf.global_variables_initializer())

            batches = bp.get_batches()

            m = np.zeros([len(items_train), 4096])

            k = 0
            while True:
                feed_dict = next(batches)
                if feed_dict is None:
                    break

                result = session.run(output, {model.t_images: feed_dict["images"], model.t_labels: feed_dict["labels"]})

                m[k: k + BATCH_SIZE] = result
                k += BATCH_SIZE

            print("Starting SVD")
            print(m.shape)
            mean = np.mean(m, 0)
            print(mean.shape)

            np.save("temp/mean.np", mean)

            m -= mean

            U, s, V = np.linalg.svd(m, full_matrices=False)

            print(U.shape)
            U = U[0:100, :]
            print(U.shape)
            np.save("temp/U", U)

p = PCA()
p.run()
