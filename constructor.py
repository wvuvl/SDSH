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
"""Net constructors"""

import tensorflow as tf
from matconvnet2tf import MatConvNet2TF


def net(batch_size, hash_size, margin=0, weight_decay_factor=0, loss_func=None):
    t_images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    t_labels = tf.placeholder(tf.int32, [None, 1])
    model = MatConvNet2TF("data/imagenet-vgg-f.mat", input=t_images, ignore=['fc8', 'prob'], do_debug_print=True)

    model.t_images = t_images
    model.t_labels = t_labels

    fcw = tf.get_variable(name='fc8_custom/weights', shape=[4096, hash_size],
                          initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                          dtype=tf.float32)
    model.weight_decay_losses.append(tf.nn.l2_loss(fcw))
    fcb = tf.get_variable(name='fc8_custom/biases', shape=[hash_size],
                          initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                          dtype=tf.float32)
    fc8 = tf.nn.bias_add(tf.matmul(model.net['relu7'], fcw), fcb)
    model.output = model.net['fc8_custom'] = fc8
    weight_decay = tf.add_n(model.weight_decay_losses)

    embedding_norm = tf.nn.l2_normalize(model.output, 1)

    model.embedding_var = tf.Variable(tf.zeros((batch_size, hash_size), dtype=tf.float32),
                                      trainable=False,
                                      name='embedding',
                                      dtype='float32')
    model.assignment = tf.assign(model.embedding_var, embedding_norm)

    if loss_func is not None:
        model.weight_decay = weight_decay * weight_decay_factor
        model.loss = loss_func(model.output, t_labels, hash_size, batch_size, margin) + model.weight_decay

    return model
