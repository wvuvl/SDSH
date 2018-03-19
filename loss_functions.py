# Copyright 2017 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Collection of loss-functions"""

import numpy as np
import tensorflow as tf
import math

def __get_triplets(batch_size, ids, boolean_mask):
    """Return boolean 3D Tensor of valid pairs mask"""
    if boolean_mask is None:
        positive_pairs = tf.equal(ids, tf.reshape(ids, [batch_size]))
    else:
        positive_pairs = boolean_mask
    eye = np.logical_not(np.eye(batch_size, dtype=np.bool))
    eye = tf.constant(eye, dtype=tf.bool)
    #positive_pairs = tf.logical_and(positive_pairs, eye)
    negative_triples = tf.logical_not(tf.reshape(positive_pairs, [batch_size, 1, batch_size]))
    triplets = tf.cast(tf.logical_and(negative_triples, positive_pairs), tf.float32)
    return triplets


def loss_accv(embedding, indices_q, indices_p, indices_n, hash_size=24, batch_size=128, margin=1.0):
    """Loss from the paper\"Deep Supervised Hashing with Triplet Labels. Xiofang Wang, Yi Shi, Kris M. Kitani\""""
    with tf.name_scope('loss') as scope:
        q = tf.gather(embedding, indices_q)
        p = tf.gather(embedding, indices_p)
        n = tf.gather(embedding, indices_n)

        bibj = tf.reduce_sum(tf.multiply(q, p), 1)
        distance_qp = bibj / 2.0

        bibj = tf.reduce_sum(tf.multiply(q, n), 1)
        distance_qn = bibj / 2.0

        dqmp = distance_qp - distance_qn

        arg = dqmp - margin

        basic_loss = tf.maximum(-arg, 0) + tf.log(1.0 + tf.exp(-tf.abs(arg)))

        binarization_regularizer = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sign(embedding) - embedding), 1))

        # From original implementation:
        # Batch size = 128
        # c = 10
        # triplets per sample = M1 * M2 * (c - 1) = 50 * 50 * 9 = 22500
        # eta = 100, but used with additional multiplier 2
        # 2 * eta * N / T / N = 2 * 100 / 22500 = 0.00888

        loss = tf.reduce_mean(basic_loss) + 0.00888 * binarization_regularizer
        return loss, basic_loss


def loss_accv_mod(embedding, indices_q, indices_p, indices_n, hash_size=24, batch_size=128, margin=1.0):
    """Modified loss from the paper
    \"Deep Supervised Hashing with Triplet Labels. Xiofang Wang, Yi Shi, Kris M. Kitani\"
    Removed binarization regularizer
    """
    embedding_norm = tf.nn.l2_normalize(embedding, 1)
    with tf.name_scope('loss') as scope:
        q = tf.gather(embedding, indices_q)
        p = tf.gather(embedding, indices_p)
        n = tf.gather(embedding, indices_n)

        bibj = tf.reduce_sum(tf.multiply(q, p), 1)
        distance_qp = bibj / 2.0

        bibj = tf.reduce_sum(tf.multiply(q, n), 1)
        distance_qn = bibj / 2.0

        dqmp = distance_qp - distance_qn

        arg = (dqmp - margin)
        arg *= hash_size

        basic_loss = tf.maximum(-arg, 0) + tf.log(1.0 + tf.exp(-tf.abs(arg)))
        loss = tf.reduce_mean(basic_loss) / hash_size
        return loss, basic_loss


def loss_triplet(embedding, indices_q, indices_p, indices_n, hash_size=24, batch_size=128, margin=1.0):
    """Regular triplet loss
    Removed binarization regularizer
    """
    embedding_norm = tf.nn.l2_normalize(embedding, 1)
    with tf.name_scope('loss') as scope:
        q = tf.gather(embedding_norm, indices_q)
        p = tf.gather(embedding_norm, indices_p)
        n = tf.gather(embedding_norm, indices_n)

        # squared euclidian distance
        # ||bi - bj|| = (bi - bj)^2 = bi^2 - 2 * bj^2 = 2 - 2 * bi * bj
        # bi^2 == 1 and because bi are normalized
        bibj = tf.reduce_sum(tf.multiply(q, p), 1)
        distance_qp = -2.0 * bibj

        bibj = tf.reduce_sum(tf.multiply(q, n), 1)
        distance_qn = -2.0 * bibj

        dqmp = distance_qp - distance_qn

        basic_loss = tf.maximum(dqmp + margin, 0.0)
        loss = tf.reduce_mean(basic_loss)
        return loss, basic_loss


def loss_spring(embedding, indices_q, indices_p, indices_n, hash_size=24, batch_size=128, margin=1.0):
    """"Energy based spring loss"""
    embedding_norm = tf.nn.l2_normalize(embedding, 1)
    with tf.name_scope('loss') as scope:
        # squared euclidian distance
        # ||bi - bj|| = (bi - bj)^2 = bi^2 - 2 * bj^2 = 2 - 2 * bi * bj
        # bi^2 == 1 and because bi are normalized

        q = tf.gather(embedding_norm, indices_q)
        p = tf.gather(embedding_norm, indices_p)
        n = tf.gather(embedding_norm, indices_n)

        print(embedding_norm.get_shape())

        bibj = tf.reduce_sum(tf.multiply(q, p), 1)
        distance_qp = -2.0 * bibj

        bibj = tf.reduce_sum(tf.multiply(q, n), 1)
        distance_qn = -2.0 * bibj

        dqmp = distance_qp - distance_qn

        # strain
        # max_distance - distance. max_distance == 2
        """
        epsilon = 1e-6
        d = tf.sqrt(4.0 - dqmp + epsilon)
        twol = 8**0.5
        K = 6
        alpha = margin
        Kp = K / (1.0-alpha**2/(twol+alpha)**2)
        C = K-Kp
        print("Kp = " + str(Kp))
        print("C = " + str(C))

        E = triplets * (Kp*tf.square(twol+alpha-d)/(twol+alpha)**2 + C)

        """
        print(dqmp.get_shape())

        epsilon = 1e-8
        d = tf.sqrt(4.0 - dqmp + epsilon)
        twol = 8**0.5

        E = tf.square(twol - d)

        loss = tf.reduce_mean(E)
        return loss, E

def loss_simplespring(embedding, indices_q, indices_p, indices_n, hash_size=24, batch_size=128, margin=1.0):
    """"Energy based spring loss"""
    embedding_norm = tf.nn.l2_normalize(embedding, 1)
    with tf.name_scope('loss') as scope:
        q = tf.gather(embedding_norm, indices_q)
        p = tf.gather(embedding_norm, indices_p)
        n = tf.gather(embedding_norm, indices_n)

        bibj = tf.matmul(embedding_norm, embedding_norm, transpose_b=True)

        # squared euclidian distance
        # ||bi - bj|| = (bi - bj)^2 = bi^2 - 2 * bj^2 = 2 - 2 * bi * bj
        # bi^2 == 1 and because bi are normalized
        distance = 2.0 - 2.0 * bibj
        negative_pairs = tf.cast(tf.logical_not(boolean_mask), tf.float32)

        # strain
        # max_distance - distance. max_distance == 2
        """
        epsilon = 1e-6
        d = tf.sqrt(4.0 - dqmp + epsilon)
        twol = 8**0.5
        K = 6
        alpha = margin
        Kp = K / (1.0-alpha**2/(twol+alpha)**2)
        C = K-Kp
        print("Kp = " + str(Kp))
        print("C = " + str(C))

        E = triplets * (Kp*tf.square(twol+alpha-d)/(twol+alpha)**2 + C)

        """
        epsilon = 1e-3
        d = tf.sqrt(distance + epsilon) - math.sqrt(epsilon)

        E = negative_pairs * tf.square(2.0 - d)# + (1.0 - negative_pairs) * distance

        loss = tf.reduce_mean(E)
        return loss, E

"""
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
"""

losses = {
    "loss_accv" : loss_accv,
    "loss_accv_mod" : loss_accv_mod,
    "loss_triplet" : loss_triplet,
    "marginloss" : loss_triplet,
    "loss_spring" : loss_spring,
    "loss_simplespring" : loss_simplespring,
}
