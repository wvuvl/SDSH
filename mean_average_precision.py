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
from utils import timer

try:
    import pyximport
    pyximport.install(setup_args={'include_dirs': np.get_include()})
    import _mean_average_precision
    has_cython = True
except:
    has_cython = False


#@timer.timer
def compute_map(hashes_train, hashes_test, labels_train, labels_test, top_n=0, and_mode=False, force_slow=False):
    """Compute MAP for given set of hashes and labels"""
    order = calc_hamming_rank(hashes_train, hashes_test, force_slow)

    if has_cython and not force_slow:
        return _mean_average_precision.calc_map(order, labels_train, labels_test, top_n, and_mode)
    else:
        print("Warning. Using slow \"compute_map\"")
        s = __compute_s(labels_train, labels_test, and_mode)
        return __calc_map(order, np.transpose(s), top_n)


def __compute_s(train_l, test_l, and_mode):
    """Return similarity matrix between two label vectors
    The output is binary matrix of size n_train x n_test
    """
    if and_mode:
        return np.bitwise_and(train_l, np.transpose(test_l)).astype(dtype=np.bool)
    else:
        return np.equal(train_l, np.transpose(test_l))


def __calc_map(order, s, top_n):
    """compute mean average precision (MAP)"""
    Q, N = s.shape
    if top_n == 0:
        top_n = N
    pos = np.asarray(range(1, top_n + 1), dtype=np.float32)
    map = 0
    for q in range(Q):
        relevance = s[q, order[q, :top_n]].astype(np.float32)
        cumulative = np.cumsum(relevance)
        number_of_relative_docs = cumulative[-1:]
        if number_of_relative_docs != 0:
            precision = cumulative / pos
            ap = np.dot(precision, relevance) / number_of_relative_docs
            map += ap
    map /= Q
    return float(map)
