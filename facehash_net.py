#! python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

import scipy.misc
import scipy.io


class vgg_f:
    def __init__(self, data_path, current):
        data = scipy.io.loadmat(data_path)
        layers = (
            'conv1', 'relu1', 'norm1', 'pool1',

            'conv2', 'relu2', 'norm2', 'pool2',

            'conv3', 'relu3', 'conv4', 'relu4',
            'conv5', 'relu5', 'pool5',
            'fc6', 'relu6', 'fc7', 'relu7'

        )
        weights = data['layers'][0]
        self.mean = np.reshape(data['normalization'][0][0][0], [1, 224, 224, 3])
        self.net = {}
        self.weight_decay_losses = []

        self.current = current - self.mean
        current = self.current
        for i, name in enumerate(layers):
            if name.startswith('conv'):
                kernels, bias = weights[i][0][0][0][0]
                bias = bias.reshape(-1)
                pad = weights[i][0][0][1]
                stride = weights[i][0][0][4]
                current = self._conv_layer(current, kernels, bias, pad, stride, i)
            elif name.startswith('relu'):
                current = tf.nn.relu(current)
            elif name.startswith('pool'):
                stride = weights[i][0][0][1]
                pad = weights[i][0][0][2]
                area = weights[i][0][0][5]
                current = self._pool_layer(current, stride, pad, area)
            elif name.startswith('fc'):
                kernels, bias = weights[i][0][0][0][0]
                bias = bias.reshape(-1)
                current = self._full_conv(current, kernels, bias, i)
            elif name.startswith('norm'):
                current = tf.nn.local_response_normalization(current, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)

            self.net[name] = current

        fcw = tf.get_variable(name='fc8/weights', shape=[4096, 24],
                              initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                              dtype=tf.float32)
        self.weight_decay_losses.append(tf.nn.l2_loss(fcw))

        fcb = tf.get_variable(name='fc8/biases', shape=[24],
                              initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                              dtype=tf.float32)

        fc8 = tf.nn.bias_add(tf.matmul(tf.reshape(self.net['relu7'], [-1, 4096]), fcw), fcb)
        self.net['fc8'] = fc8
        self.weight_decay = tf.add_n(self.weight_decay_losses)

    def _conv_layer(self, input, weights, bias, pad, stride, i):
        pad = pad[0]
        stride = stride[0]
        input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
        w = tf.Variable(weights, name='w' + str(i), dtype='float32')
        b = tf.Variable(bias, name='bias' + str(i), dtype='float32')
        self.weight_decay_losses.append(tf.nn.l2_loss(w))
        self.net['weights' + str(i)] = w
        self.net['b' + str(i)] = b
        conv = tf.nn.conv2d(input, w, strides=[1, stride[0], stride[1], 1], padding='VALID', name='conv' + str(i))
        return tf.nn.bias_add(conv, b, name='add' + str(i))

    def _full_conv(self, input, weights, bias, i):
        w = tf.Variable(weights, name='w' + str(i), dtype='float32')
        b = tf.Variable(bias, name='bias' + str(i), dtype='float32')
        self.weight_decay_losses.append(tf.nn.l2_loss(w))
        self.net['weights' + str(i)] = w
        self.net['b' + str(i)] = b
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='VALID', name='fc' + str(i))
        return tf.nn.bias_add(conv, b, name='add' + str(i))

    def _pool_layer(self, input, stride, pad, area):
        pad = pad[0]
        area = area[0]
        stride = stride[0]
        input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
        return tf.nn.max_pool(input, ksize=[1, area[0], area[1], 1], strides=[1, stride[0], stride[1], 1], padding='VALID')


def loss_accv(embedding, ids, HASH_SIZE=24, BATCH_SIZE=128):
    #embedding_norm = tf.nn.l2_normalize(embedding, 1)
    with tf.name_scope('loss') as scope:
        bibj = tf.matmul(embedding, embedding, transpose_b=True)
        distance = bibj / 2.0
        negative_distance = tf.reshape(distance, [BATCH_SIZE, 1, BATCH_SIZE])

        sim = tf.equal(ids, tf.reshape(ids, [BATCH_SIZE]))
        s = tf.cast(sim, tf.int32) * ids
        striple = tf.cast(tf.logical_and(tf.not_equal(s, tf.reshape(ids, [BATCH_SIZE, 1, 1])), sim), tf.float32)

        arg = distance - negative_distance - HASH_SIZE / 2.0

        basic_loss = tf.maximum(-arg, 0) + tf.log(1.0 + tf.exp(-tf.abs(arg)))
        loss = tf.reduce_sum(striple * basic_loss) / 303750.0 + 0.00888 * tf.reduce_mean(tf.reduce_sum(tf.square(tf.sign(embedding) - embedding), 1))
        return loss


def loss_accv_mod(embedding, ids, HASH_SIZE=24, BATCH_SIZE=128, MARGIN=0.5):
    embedding_norm = tf.nn.l2_normalize(embedding, 1)
    with tf.name_scope('loss') as scope:
        bibj = tf.matmul(embedding_norm, embedding_norm, transpose_b=True)
        distance = bibj / 2.0
        negative_distance = tf.reshape(distance, [BATCH_SIZE, 1, BATCH_SIZE])

        sim = tf.equal(ids, tf.reshape(ids, [BATCH_SIZE]))
        s = tf.cast(sim, tf.int32) * ids
        striple = tf.cast(tf.logical_and(tf.not_equal(s, tf.reshape(ids, [BATCH_SIZE, 1, 1])), sim), tf.float32)

        arg = 24 * (distance - negative_distance - MARGIN)

        basic_loss = tf.maximum(-arg, 0) + tf.log(1.0 + tf.exp(-tf.abs(arg)))
        loss = tf.reduce_sum(striple * basic_loss) / 303750.0 / 24.0
        return loss


def loss(embedding, ids, HASH_SIZE=24, BATCH_SIZE=128, MARGIN=1.0):
    embedding_norm = tf.nn.l2_normalize(embedding, 1)
    with tf.name_scope('loss') as scope:
        bibj = tf.matmul(embedding_norm, embedding_norm, transpose_b=True)
        distance = 2.0 - 2.0 * bibj
        negative_distance = tf.reshape(distance, [BATCH_SIZE, 1, BATCH_SIZE])

        sim = tf.equal(ids, tf.reshape(ids, [BATCH_SIZE]))

        #striple = tf.cast(
        #    tf.logical_and(sim, tf.logical_not(tf.reshape(sim, [BATCH_SIZE, 1, BATCH_SIZE]))),
        #    tf.float32) * tf.constant(np.ones((BATCH_SIZE, BATCH_SIZE), dtype=np.float32) - np.eye(BATCH_SIZE, dtype=np.float32))
        #print(striple.shape)

        s = tf.cast(sim, tf.int32) * ids
        striple = tf.cast(tf.logical_and(tf.not_equal(s, tf.reshape(ids, [BATCH_SIZE, 1, 1])), sim), tf.float32)

        basic_loss = striple * tf.maximum(distance - negative_distance + MARGIN, 0.0)
        l = tf.reduce_mean(basic_loss)
        tf.summary.scalar('contrustive_loss', l)
        return l

def loss_spring(embedding, ids, HASH_SIZE=24, BATCH_SIZE=128, MARGIN=1.0):
    embedding_norm = tf.nn.l2_normalize(embedding, 1)
    with tf.name_scope('loss') as scope:
        bibj = tf.matmul(embedding_norm, embedding_norm, transpose_b=True)
        distance = 2.0 - 2.0 * bibj
        negative_distance = tf.reshape(distance, [BATCH_SIZE, 1, BATCH_SIZE])

        sim = tf.equal(ids, tf.reshape(ids, [BATCH_SIZE]))

        s = tf.cast(sim, tf.int32) * ids
        striple = tf.cast(tf.logical_and(tf.not_equal(s, tf.reshape(ids, [BATCH_SIZE, 1, 1])), sim), tf.float32)

        d = tf.sqrt(-distance + negative_distance + 4.0 + 1e-6)
        twol = 8**0.5
        K = 6
        alpha = MARGIN
        Kp = K / (1.0-alpha**2/(twol+alpha)**2)
        C = K-Kp
        print("Kp = " + str(Kp))
        print("C = " + str(C))

        E = striple * (Kp*tf.square(twol+alpha-d)/(twol+alpha)**2 + C)

        l = tf.reduce_sum(E) / 303750.0
        tf.summary.scalar('spring_loss', l)
        return l

def loss2(embedding, ids, HASH_SIZE=24, BATCH_SIZE=128):
    embedding_norm = tf.nn.l2_normalize(embedding, 1)

    with tf.name_scope('loss') as scope:
        ids = tf.reshape(ids, [-1, 1])

        with tf.name_scope('diameter_loss') as scope:
            template = tf.reshape(tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), [1, 10])
            template = tf.cast(tf.equal(ids, template), tf.float32)

            table = tf.reshape(template, [-1, 10, 1]) * tf.reshape(embedding_norm, [-1, 1, HASH_SIZE])


            centers = tf.nn.l2_normalize(tf.reshape(tf.reduce_mean(table, 0), [10, HASH_SIZE]), 1)

            table_centered = table - tf.reshape(centers, [1, 10, HASH_SIZE])

            distances = tf.reduce_sum(tf.square(tf.norm(table_centered + 1e-6, axis=2)) * tf.reshape(template, [-1, 10]), 0)

            tf.summary.image('distances', tf.reshape(distances, [1, 1, -1, 1]))

            width = tf.reduce_sum(distances)
            width /= BATCH_SIZE

            diameter_loss = width

        with tf.name_scope('separation_loss') as scope:
            centers = tf.reshape(centers, [10, HASH_SIZE])
            bibj = tf.matmul(centers, centers, transpose_b=True)
            distance = tf.pow(0.5 + 1e-6 - bibj / 2.0, 0.5)

            energy = tf.pow(1.0 - distance, 2.0)
            tf.summary.image('disimilarity', tf.reshape(energy, [1, 10, 10, 1]))

            separation_loss = tf.reduce_mean(energy * tf.constant(np.ones((10, 10), dtype=np.float32) - np.eye(10, dtype=np.float32)))

        diameter_loss *= 5.0

        tf.summary.scalar('diameter_loss', diameter_loss)
        tf.summary.scalar('separation_loss', separation_loss)

        total = diameter_loss + separation_loss
        tf.summary.scalar('total_loss', total)
        return diameter_loss + separation_loss
