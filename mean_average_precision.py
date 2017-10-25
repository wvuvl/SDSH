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

import numpy as np

from utils.hamming import calc_hamming_rank


def compute_map(hashes_train, hashes_test, labels_train, labels_test):
    """Compute MAP for given set of hashes and labels"""
    order_h = calc_hamming_rank(hashes_train, hashes_test)
    s = __compute_s(labels_train, labels_test)
    return __calc_map(order_h, np.transpose(s))


def __compute_s(train_l, test_l):
    """Return similarity matrix between two label vectors
    The output is binary matrix of size n_train x n_test
    """
    d = train_l - np.transpose(test_l)
    return np.equal(d, 0)


def __calc_map(order_h, neighbor):
    """compute mean average precision (MAP)"""
    (Q, N) = neighbor.shape
    pos = np.asarray(range(1, N + 1))
    map = 0
    num_succ = 0
    for i in range(Q):
        ngb = neighbor[i, order_h[i, :]]
        n_rel = np.sum(ngb)
        if n_rel > 0:
            prec = np.cumsum(ngb) / pos
            ap = np.mean(prec[ngb])
            map += ap
            num_succ += 1

    map /= num_succ
    num_succ /= Q
    return map, num_succ
