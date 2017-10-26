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
"""Faster map using cython"""

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef __calc_map(np.int32_t[:,::1] order, np.int8_t[:,::1] labels_train, np.int8_t[:,::1] labels_test):
    """compute mean average precision (MAP)"""

    cdef np.float32_t map = <float>0.0

    cdef np.intp_t Q = order.shape[0]
    cdef np.intp_t N = order.shape[1]

    cdef np.ndarray[np.float32_t, ndim=1] pos = np.arange(1, N + 1, dtype=np.float32)

    cdef np.float32_t[::1] relevance = np.zeros(N, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] precision
    cdef np.ndarray[np.float32_t, ndim=1] cumulative
    cdef np.float32_t ap
    cdef np.float32_t number_of_relative_docs

    cdef int index

    for q in range(Q):
        for i in range(N):
            index = order[q, i]
            relevance[i] = <float>1.0 if labels_test[q, 0] == labels_train[index, 0] else <float>0.0
        cumulative = np.cumsum(relevance)
        number_of_relative_docs = cumulative[N-1]
        precision = cumulative / pos
        ap = np.dot(precision, relevance) / number_of_relative_docs
        map += ap
    map /= Q
    return map

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_map(order, labels_train, labels_test):
    return __calc_map(order.astype(np.int32), labels_train.astype(np.int8), labels_test.astype(np.int8))
