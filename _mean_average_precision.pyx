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
from utils.timer import timer
from hashranking import hamming

cdef extern int __builtin_popcount(unsigned int) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef __calc_map(np.int32_t[:,::1] order, np.int32_t[:,::1] labels_train, np.int32_t[:,::1] labels_test, int top_n):
    """compute mean average precision (MAP)"""

    cdef np.float32_t map = <float>0.0

    cdef np.intp_t Q = order.shape[0]
    cdef np.intp_t N = order.shape[1]

    if top_n == 0:
        top_n = N

    cdef np.ndarray[np.float32_t, ndim=1] pos = np.arange(1, top_n + 1, dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=1] av_precision = np.zeros(top_n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] av_recall = np.zeros(top_n, dtype=np.float32)

    cdef np.float32_t[::1] relevance = np.zeros(top_n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] precision
    cdef np.ndarray[np.float32_t, ndim=1] recall
    cdef np.ndarray[np.float32_t, ndim=1] cumulative
    cdef np.float32_t ap
    cdef np.float32_t number_of_relative_docs
    cdef np.float32_t total_number_of_relevant_documents

    cdef int index

    for q in range(Q):
        for i in range(top_n):
            index = order[q, i]
            relevance[i] = <float>1.0 if labels_test[q, 0] == labels_train[index, 0] else <float>0.0

        cumulative = np.cumsum(relevance)
        number_of_relative_docs = cumulative[top_n-1]

        total_number_of_relevant_documents = 0

        for i in range(N):
            index = order[q, i]
            total_number_of_relevant_documents += <float>1.0 if labels_test[q, 0] == labels_train[index, 0] else <float>0.0

        if number_of_relative_docs != 0:
            precision = cumulative / pos
            recall = cumulative / total_number_of_relevant_documents
            av_precision += precision
            av_recall += recall
            ap = np.dot(precision, relevance) / number_of_relative_docs
            map += ap
    map /= Q
    av_precision /= Q
    av_recall /= Q

    cdef np.ndarray[np.float32_t, ndim=2] curve = np.zeros([top_n, 2], dtype=np.float32)
    curve[:, 0] = av_precision
    curve[:, 1] = av_recall

    return map, curve



#@timer
@cython.boundscheck(False)
@cython.wraparound(False)
cdef __calc_map_and(np.int32_t[:, ::1] order, np.uint64_t[::1] labels_trainL, np.uint64_t[::1] labels_trainH, np.uint64_t[::1] labels_testL, np.uint64_t[::1] labels_testH, int top_n):
    """compute mean average precision (MAP)"""

    cdef np.float32_t map = <float>0.0

    cdef np.intp_t Q = order.shape[0]
    cdef np.intp_t N = order.shape[1]

    if top_n == 0:
        top_n = N

    cdef np.ndarray[np.float32_t, ndim=1] pos = np.arange(1, top_n + 1, dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=1] av_precision = np.zeros(top_n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] av_recall = np.zeros(top_n, dtype=np.float32)

    cdef np.float32_t[::1] relevance = np.zeros(top_n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] precision
    cdef np.ndarray[np.float32_t, ndim=1] recall
    cdef np.ndarray[np.float32_t, ndim=1] cumulative
    cdef np.float32_t ap
    cdef np.float32_t number_of_relative_docs
    cdef np.float32_t total_number_of_relevant_documents

    cdef int index

    for q in range(Q):
        for i in range(top_n):
            index = order[q, i]
            relevance[i] = <float>1.0 if ((labels_testL[q] & labels_trainL[index]) | (labels_testH[q] & labels_trainH[index])) != <unsigned int>0 else <float>0.0
        cumulative = np.cumsum(relevance)
        number_of_relative_docs = cumulative[top_n-1]

        total_number_of_relevant_documents = 0

        for i in range(N):
            index = order[q, i]
            total_number_of_relevant_documents += <float>1.0 if ((labels_testL[q] & labels_trainL[index]) | (labels_testH[q] & labels_trainH[index])) != <unsigned int>0 else <float>0.0

        if number_of_relative_docs != 0:
            precision = cumulative / pos
            recall = cumulative / total_number_of_relevant_documents
            av_precision += precision
            av_recall += recall
            ap = np.dot(precision, relevance) / number_of_relative_docs
            map += ap
    map /= Q
    av_precision /= Q
    av_recall /= Q

    cdef np.ndarray[np.float32_t, ndim=2] curve = np.zeros([top_n, 2], dtype=np.float32)
    curve[:, 0] = av_precision
    curve[:, 1] = av_recall

    return map, curve




@cython.boundscheck(False)
@cython.wraparound(False)
cdef __calc_map_weighted(np.int32_t[:,::1] order,np.int32_t[:,::1] labels_train,np.int32_t[:,::1] labels_test):

    cdef np.float32_t map = <float>0.0

    cdef np.intp_t Q = order.shape[0]
    cdef np.intp_t N = order.shape[1]

    cdef np.intp_t relCount = 0
    cdef np.float32_t ap = <float>0.0

    cdef int index
    cdef int rel
    cdef np.float32_t acg



    for q in range(Q):
        relCount = 0
        ap = <float>0.0
        for i in range(100):
            index = order[q,i]
            rel = __builtin_popcount(labels_train[index,0]&labels_test[q,0])

            if (rel > 0):
                relCount += 1
                acg = rel
                for n in range(i-1,0,-1):
                    index = order[q,n]
                    acg += __builtin_popcount(labels_train[index,0]&labels_test[q,0])
                acg/= i+1

                ap+= acg
        if relCount > 0:
            ap /= relCount
            map += ap
    map /= Q
    return map,None



@cython.boundscheck(False)
@cython.wraparound(False)
def labeles_to_two_64bword(np.ndarray labels):
    cdef np.intp_t N = labels.shape[0]
    cdef np.uint64_t[::1] outh = np.zeros(N, dtype=np.uint64)
    cdef np.uint64_t[::1] outl = np.zeros(N, dtype=np.uint64)
    for i in range(N):
        outh[i] = (np.uint64)((labels[i] >> 64) & 0xFFFFFFFFFFFFFFFF)
        outl[i] = (np.uint64)((labels[i]) & 0xFFFFFFFFFFFFFFFF)
    return outl, outh


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_map(order, labels_train, labels_test, top_n, and_mode,weighted_mode = False):

    if and_mode:
        labels_trainL, labels_trainH = labeles_to_two_64bword(labels_train)
        labels_testL, labels_testH = labeles_to_two_64bword(labels_test)
        return __calc_map_and(order.astype(np.int32), labels_trainL, labels_trainH, labels_testL, labels_testH, top_n)
    elif not weighted_mode:
        return __calc_map(order.astype(np.int32), labels_train.astype(np.int32), labels_test.astype(np.int32), top_n)
    else:
        return __calc_map_weighted(order.astype(np.int32),labels_train.astype(np.int32),labels_test.astype(np.int32))

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_map_fast(hashes_train, hashes_test, labels_train, labels_test, and_mode,weighted_mode=False):
    hr = hamming.HashRankingContext()
    hr.Init(hashes_train.shape[0], hashes_test.shape[0], 1 if hashes_train.shape[1] > 32 else 0, 1 if and_mode else 0 if not weighted_mode else 2)

    hr.LoadQueryHashes(hashes_test)
    hr.LoadDBHashes(hashes_train)

    if and_mode:
        labels_trainL, labels_trainH = labeles_to_two_64bword(labels_train)
        labels_testL, labels_testH = labeles_to_two_64bword(labels_test)
        hr.LoadQueryLabelsLDW(labels_testL)
        hr.LoadQueryLabelsHDW(labels_testH)
        hr.LoadDBLabelsLDW(labels_trainL)
        hr.LoadDBLabelsHDW(labels_trainH)
    else:
        hr.LoadQueryLabels(np.array(labels_test).flatten().astype(np.int32))
        hr.LoadDBLabels(np.array(labels_train).flatten().astype(np.int32))

    return hr.Map()
