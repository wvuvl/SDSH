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
import numpy as np

def net(batch_size, hash_size, expected_triplet_count=100, margin=0, weight_decay_factor=0, loss_func=None):
    t_images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    t_latent = tf.placeholder(tf.float32, [None, 9216])
    t_labels = tf.placeholder(tf.int32, [None, 1])
    t_boolmask = tf.placeholder(tf.bool, [batch_size, batch_size])
    t_indices_q = tf.placeholder(tf.int32, [expected_triplet_count])
    t_indices_p = tf.placeholder(tf.int32, [expected_triplet_count])
    t_indices_n = tf.placeholder(tf.int32, [expected_triplet_count])

    if True:
        model = MatConvNet2TF("data/imagenet-vgg-f_old.mat", input=t_images, ignore=['fc8', 'prob'], do_debug_print=True, input_latent=t_latent, latent_layer="fc6")
    else:
        class Model:
            def __init__(self, input=None):
                self.deep_param_img = {}
                self.train_layers = []
                self.train_last_layer = []
                self.net = {}
                self.weight_decay_losses = []
                print("loading img model")
                net_data = np.load('reference_pretrain.npy', encoding='bytes').item()

                # swap(2,1,0)
                reshaped_image = input
                tm = tf.Variable([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=tf.float32)
                reshaped_image = tf.reshape(reshaped_image, [-1, 3])
                reshaped_image = tf.matmul(reshaped_image, tm)
                reshaped_image = tf.reshape(reshaped_image, [-1, 227, 227, 3])

                ### Zero-mean input
                with tf.name_scope('preprocess') as scope:
                    mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img-mean')
                    distorted_image = reshaped_image - mean

                ### Conv1
                ### Output 96, kernel 11, stride 4
                with tf.name_scope('conv1') as scope:
                    kernel = tf.Variable(net_data['conv1'][0], name='weights')
                    conv = tf.nn.conv2d(distorted_image, kernel, [1, 4, 4, 1], padding='VALID')
                    biases = tf.Variable(net_data['conv1'][1], name='biases')
                    out = tf.nn.bias_add(conv, biases)
                    self.conv1 = tf.nn.relu(out, name=scope)
                    self.deep_param_img['conv1'] = [kernel, biases]
                    self.train_layers += [kernel, biases]

                ### Pool1
                self.pool1 = tf.nn.max_pool(self.conv1,
                                            ksize=[1, 3, 3, 1],
                                            strides=[1, 2, 2, 1],
                                            padding='VALID',
                                            name='pool1')

                ### LRN1
                radius = 2
                alpha = 2e-05
                beta = 0.75
                bias = 1.0
                self.lrn1 = tf.nn.local_response_normalization(self.pool1,
                                                               depth_radius=radius,
                                                               alpha=alpha,
                                                               beta=beta,
                                                               bias=bias)

                ### Conv2
                ### Output 256, pad 2, kernel 5, group 2
                with tf.name_scope('conv2') as scope:
                    kernel = tf.Variable(net_data['conv2'][0], name='weights')
                    group = 2
                    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
                    input_groups = tf.split(self.lrn1, group, 3)
                    kernel_groups = tf.split(kernel, group, 3)
                    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                    ### Concatenate the groups
                    conv = tf.concat(output_groups, 3)

                    biases = tf.Variable(net_data['conv2'][1], name='biases')
                    out = tf.nn.bias_add(conv, biases)
                    self.conv2 = tf.nn.relu(out, name=scope)
                    self.deep_param_img['conv2'] = [kernel, biases]
                    self.train_layers += [kernel, biases]

                ### Pool2
                self.pool2 = tf.nn.max_pool(self.conv2,
                                            ksize=[1, 3, 3, 1],
                                            strides=[1, 2, 2, 1],
                                            padding='VALID',
                                            name='pool2')

                ### LRN2
                radius = 2
                alpha = 2e-05
                beta = 0.75
                bias = 1.0
                self.lrn2 = tf.nn.local_response_normalization(self.pool2,
                                                               depth_radius=radius,
                                                               alpha=alpha,
                                                               beta=beta,
                                                               bias=bias)

                ### Conv3
                ### Output 384, pad 1, kernel 3
                with tf.name_scope('conv3') as scope:
                    kernel = tf.Variable(net_data['conv3'][0], name='weights')
                    conv = tf.nn.conv2d(self.lrn2, kernel, [1, 1, 1, 1], padding='SAME')
                    biases = tf.Variable(net_data['conv3'][1], name='biases')
                    out = tf.nn.bias_add(conv, biases)
                    self.conv3 = tf.nn.relu(out, name=scope)
                    self.deep_param_img['conv3'] = [kernel, biases]
                    self.train_layers += [kernel, biases]

                ### Conv4
                ### Output 384, pad 1, kernel 3, group 2
                with tf.name_scope('conv4') as scope:
                    kernel = tf.Variable(net_data['conv4'][0], name='weights')
                    group = 2
                    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
                    input_groups = tf.split(self.conv3, group, 3)
                    kernel_groups = tf.split(kernel, group, 3)
                    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                    ### Concatenate the groups
                    conv = tf.concat(output_groups, 3)
                    biases = tf.Variable(net_data['conv4'][1], name='biases')
                    out = tf.nn.bias_add(conv, biases)
                    self.conv4 = tf.nn.relu(out, name=scope)
                    self.deep_param_img['conv4'] = [kernel, biases]
                    self.train_layers += [kernel, biases]

                ### Conv5
                ### Output 256, pad 1, kernel 3, group 2
                with tf.name_scope('conv5') as scope:
                    kernel = tf.Variable(net_data['conv5'][0], name='weights')
                    group = 2
                    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
                    input_groups = tf.split(self.conv4, group, 3)
                    kernel_groups = tf.split(kernel, group, 3)
                    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                    ### Concatenate the groups
                    conv = tf.concat(output_groups, 3)
                    biases = tf.Variable(net_data['conv5'][1], name='biases')
                    out = tf.nn.bias_add(conv, biases)
                    self.conv5 = tf.nn.relu(out, name=scope)
                    self.deep_param_img['conv5'] = [kernel, biases]
                    self.train_layers += [kernel, biases]

                ### Pool5
                self.pool5 = tf.nn.max_pool(self.conv5,
                                            ksize=[1, 3, 3, 1],
                                            strides=[1, 2, 2, 1],
                                            padding='VALID',
                                            name='pool5')

                self.prob = tf.placeholder_with_default(1.0, shape=())
                ### FC6
                ### Output 4096
                with tf.name_scope('fc6') as scope:
                    shape = int(np.prod(self.pool5.get_shape()[1:]))
                    fc6w = tf.Variable(net_data['fc6'][0], name='weights')
                    fc6b = tf.Variable(net_data['fc6'][1], name='biases')
                    pool5_flat = tf.reshape(self.pool5, [-1, shape])
                    self.fc5 = pool5_flat
                    fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
                    self.fc6 = tf.nn.dropout(tf.nn.relu(fc6l), self.prob)
                    self.fc6o = tf.nn.relu(fc6l)
                    self.deep_param_img['fc6'] = [fc6w, fc6b]
                    self.train_layers += [fc6w, fc6b]
                    self.weight_decay_losses.append(tf.nn.l2_loss(fc6w))

                ### FC7
                ### Output 4096
                with tf.name_scope('fc7') as scope:
                    fc7w = tf.Variable(net_data['fc7'][0], name='weights')
                    fc7b = tf.Variable(net_data['fc7'][1], name='biases')
                    fc7l = tf.nn.bias_add(tf.matmul(self.fc6, fc7w), fc7b)
                    self.fc7 = tf.nn.dropout(tf.nn.relu(fc7l), self.prob)
                    fc7lo = tf.nn.bias_add(tf.matmul(self.fc6o, fc7w), fc7b)
                    self.fc7o = tf.nn.relu(fc7lo)
                    self.deep_param_img['fc7'] = [fc7w, fc7b]
                    self.train_layers += [fc7w, fc7b]
                    self.weight_decay_losses.append(tf.nn.l2_loss(fc7w))

                    self.net['relu7'] = self.fc7o

                self.weight_decay = tf.add_n(self.weight_decay_losses)


        print("img modal loading finished")
        ### Return outputs

        model = Model(t_images)


    model.t_images = t_images
    model.t_latent = t_latent
    model.t_labels = t_labels
    model.t_boolmask = t_boolmask
    model.t_indices_q = t_indices_q
    model.t_indices_p = t_indices_p
    model.t_indices_n = t_indices_n

    fcw = tf.get_variable(name='fc8_custom/weights', shape=[4096, hash_size],
                          initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                          dtype=tf.float32)

    fcb = tf.get_variable(name='fc8_custom/biases', shape=[hash_size],
                          initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                          dtype=tf.float32)

    model.weight_decay_losses.append(tf.abs(tf.reduce_mean(tf.reduce_sum(tf.square(fcw), 0)) - 1.0))
    weight_decay = tf.add_n(model.weight_decay_losses)
    model.weight_decay = weight_decay * weight_decay_factor

    fc8 = tf.nn.bias_add(tf.matmul(model.net['relu7'], fcw), fcb)
    model.output = model.net['fc8_custom'] = fc8
    model.output_norm = tf.nn.l2_normalize(model.output, 1)

    fc8 = tf.nn.bias_add(tf.matmul(model.output2, fcw), fcb)
    model.output_2 = model.net['fc8_custom_2'] = fc8

    model.embedding_var = tf.Variable(tf.zeros((batch_size, hash_size), dtype=tf.float32),
                                      trainable=False,
                                      name='embedding',
                                      dtype='float32')
    model.assignment = tf.assign(model.embedding_var, model.output_norm)

    if loss_func is not None:
        model.loss, model.E = loss_func(model.output, t_indices_q, t_indices_p, t_indices_n, hash_size, batch_size, margin)
        model.loss_2, model.E = loss_func(model.output_2, t_indices_q, t_indices_p, t_indices_n, hash_size, batch_size, margin)

    return model
