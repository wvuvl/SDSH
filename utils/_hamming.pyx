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
"""Faster hamming distance using cython"""

import numpy as np
cimport numpy as np
cimport cython

#cdef extern from "intrin.h":
#    np.uint32_t __popcnt(np.uint32_t value);
#    np.uint64_t __popcnt64(np.uint64_t value);
cdef extern int __builtin_popcount(unsigned int) nogil
cdef extern int __builtin_popcountll(unsigned long long) nogil

cdef inline np.int8_t __hamming_distance32(np.uint32_t x, np.uint32_t y):
    cdef np.uint32_t val = x ^ y
    return __builtin_popcount(val)

cdef inline np.int8_t __hamming_distance64(np.uint64_t x, np.uint64_t y):
    cdef np.uint64_t val = x ^ y
    return __builtin_popcountll(val)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.uint32_t[::1] __to_int32_hashes(np.ndarray[np.float32_t, ndim=2] p):
    cdef np.intp_t w = p.shape[1]
    cdef np.intp_t h = p.shape[0]
    cdef np.uint32_t output = 0
    cdef np.uint32_t power = 1
    cdef np.float32_t[:, ::1] p_v = p;

    cdef np.uint32_t[::1] out = np.zeros(h, dtype=np.uint32)
    for x in range(h):
        output = 0
        power = 1
        for y in range(w):
            output += power if p_v[x, y] > 0.0 else 0
            power *= 2
        out[x] = output
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.uint64_t[::1] __to_int64_hashes(np.ndarray[np.float32_t, ndim=2] p):
    cdef np.intp_t w = p.shape[1]
    cdef np.intp_t h = p.shape[0]
    cdef np.uint64_t output = 0
    cdef np.uint64_t power = 1
    cdef np.float32_t[:, ::1] p_v = p;

    cdef np.uint64_t[::1] out = np.zeros(h, dtype=np.uint64)
    for x in range(h):
        output = 0
        power = 1
        for y in range(w):
            if p_v[x, y] > 0.0:
                output += power
            power *= 2
        out[x] = output
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_hamming_dist(b1, b2):
    """Compute the hamming distance between every pair of data points represented in each row of b1 and b2"""
    cdef np.uint32_t[::1] p1 = __to_int32_hashes(b1)
    cdef np.uint32_t[::1] p2 = __to_int32_hashes(b2)

    cdef np.intp_t l1 = p1.shape[0]
    cdef np.intp_t l2 = p2.shape[0]

    cdef np.ndarray[np.int8_t, ndim=2] out = np.zeros([l1, l2], dtype=np.int8)
    cdef np.int8_t[:, ::1] out_v = out;

    cdef np.int8_t d = 0
    cdef np.int8_t dist = 0
    cdef np.uint32_t val = 0

    for x in range(l1):
        for y in range(l2):
            out_v[x, y] = __hamming_distance32(p1[x], p2[y])

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_hamming_dist64(b1, b2):
    """Compute the hamming distance between every pair of data points represented in each row of b1 and b2"""
    cdef np.uint64_t[::1] p1 = __to_int64_hashes(b1)
    cdef np.uint64_t[::1] p2 = __to_int64_hashes(b2)

    cdef np.intp_t l1 = p1.shape[0]
    cdef np.intp_t l2 = p2.shape[0]

    cdef np.ndarray[np.int8_t, ndim=2] out = np.zeros([l1, l2], dtype=np.int8)
    cdef np.int8_t[:, ::1] out_v = out;

    for x in range(l1):
        for y in range(l2):
            out_v[x, y] = __hamming_distance64(p1[x], p2[y])

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def sort(a):
    cdef np.int8_t[:, ::1] a_v = a;
    cdef np.intp_t l1 = a.shape[0]
    cdef np.intp_t l2 = a.shape[1]

    cdef np.int32_t[65] count
    cdef np.int32_t total
    cdef np.int32_t old_count
    cdef np.int8_t key

    cdef np.ndarray[np.int32_t, ndim=2] out = np.zeros([l1, l2], dtype=np.int32)
    cdef np.int32_t[:, ::1] out_v = out;

    cdef np.int32_t[::1] tmp = np.zeros([l2], dtype=np.int32)

    for x in range(l1):
        for i in range(65):
            count[i] = 0
        for y in range(l2):
            count[a_v[x, y]] += 1
        total = 0
        old_count = 0
        for i in range(65):
            old_count = count[i]
            count[i] = total
            total += old_count

        for y in range(l2):
            key = a_v[x, y]
            tmp[y] = count[key]
            count[key] += 1

        for y in range(l2):
            out_v[x, tmp[y]] = y

    return out
