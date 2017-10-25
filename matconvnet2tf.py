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

# This code based on the code taken from https://github.com/cuixue/vgg-f-tensorflow.

import numpy as np
import scipy.io
import tensorflow as tf


class MatConvNet2TF:
    """Reads matconvnet file creates tensorflow graph."""
    def __init__(self, data_path):
        data = scipy.io.loadmat(data_path, struct_as_record=False, squeeze_me=True)
        layers = data['layers']
        self.mean = np.array(data['meta'].normalization.averageImage, ndmin=4)
        self.net = {}
        self.weight_decay_losses = []
        self.net['classes'] = data['meta'].classes.description

        input_shape = tuple(data['meta'].inputs.size[:3])
        input_shape = (None,) + input_shape
        self.input = tf.placeholder('float32', input_shape)

        self.current = self.input - self.mean
        current = self.current
        for i, layer in enumerate(layers):
            if layer.type == 'conv':
                current = self._conv_layer(current, layer)
            elif layer.type == 'relu':
                current = tf.nn.relu(current, name=layer.name)
            elif layer.type == 'pool':
                current = self._pool_layer(current, layer)
            elif layer.type == 'lrn':
                # depth_radius = (N - 1) / 2; PARAM = [N KAPPA ALPHA BETA]
                n = layer.param[0]
                current = tf.nn.local_response_normalization(current,
                                                             depth_radius=(n-1)//2,
                                                             bias=layer.param[1],
                                                             alpha=layer.param[2],
                                                             beta=layer.param[3],
                                                             name=layer.name)
            elif layer.type == 'softmax':
                current = tf.nn.softmax(current, name=layer.name)

            self.net[layer.name] = current
        self.weight_decay = tf.add_n(self.weight_decay_losses)

    @staticmethod
    def _convert_pad(pad):
        return [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]]

    @staticmethod
    def _convert_stride(stride):
        return [1, stride[0], stride[1], 1]

    def _conv_layer(self, input, layer):
        with tf.name_scope(layer.name):
            input = tf.pad(input, self._convert_pad(layer.pad), "CONSTANT")
            weights, biases = layer.weights
            weights = np.array(weights, ndmin=4)
            biases = biases.reshape(-1)
            w = tf.Variable(weights, name='weights', dtype='float32')
            b = tf.Variable(biases, name='biases', dtype='float32')
            self.weight_decay_losses.append(tf.nn.l2_loss(w))
            conv = tf.nn.conv2d(input,
                                w,
                                strides=self._convert_stride(layer.stride),
                                padding='VALID',
                                name=layer.name)
            return tf.nn.bias_add(conv, b, name='add')

    def _pool_layer(self, input, layer):
        with tf.name_scope(layer.name):
            input = tf.pad(input, self._convert_pad(layer.pad), "CONSTANT")
            return tf.nn.max_pool(input,
                                  ksize=self._convert_stride(layer.pool),
                                  strides=self._convert_stride(layer.stride),
                                  padding='VALID')
