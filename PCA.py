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

            test = False

            if test:
                output = model.net['fc8_custom']
            else:
                output = model.net['relu7']

            session.run(tf.global_variables_initializer())

            batches = bp.get_batches()

            if test:
                m = np.zeros([len(items_train), 24])
            else:
                m = np.zeros([len(items_train), 4096])

            k = 0
            while True:
                feed_dict = next(batches)
                if feed_dict is None:
                    break

                result = session.run(output, {model.t_images: feed_dict["images"], model.t_labels: feed_dict["labels"]})

                m[k: k + BATCH_SIZE] = result
                k += BATCH_SIZE

            mean = np.mean(m, 0)
            print("mean {0}", mean)
            print("std {0}", np.std(m,0))

            np.save("temp/mean", mean)
            m -= mean
            # get the covariance matrix
            Xcov = np.dot(m.T, m)
            #
            # # eigenvalue decomposition of the covariance matrix
            d, V = np.linalg.eigh(Xcov)
            #
            # # a fudge factor can be used so that eigenvectors associated with
            # # small eigenvalues do not get overamplified.
            D = np.diagflat(1.0 / np.sqrt(d[-24:]))
            #
            W = np.dot(V[:,-24:], D)
            #
            np.save("temp/W", W)
            #
            # print("Starting SVD")
            # print(m.shape)
            # mean = np.mean(m, 0)
            # print(mean.shape)
            #
            # np.save("temp/mean", mean)
            # print("mean {0}", mean)
            #
            # m -= mean
            #
            # U, s, V = np.linalg.svd(m, full_matrices=False)
            #
            # print(U.shape)
            # U = U[0:100, :]
            # print(U.shape)
            # np.save("temp/U", U)
            # np.save("temp/V", V)
            # np.save("temp/s", np.diag(s))
            # print("U {0}", U.shape)
            # print("s {0}", s.shape)
            # print("V {0}", V.shape)

p = PCA()
p.run()
