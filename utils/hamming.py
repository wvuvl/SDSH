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
"""Methods to work with hashes"""

import numpy as np

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
try:
    from utils import _hamming
    from utils.timer import timer
except:
    import _hamming
    import timer

has_cython = True


#@timer
def calc_hamming_dist(b1, b2):
    """Compute the hamming distance between every pair of data points represented in each row of b1 and b2"""
    p1 = np.sign(b1).astype(np.int8)
    p2 = np.sign(b2).astype(np.int8)

    r = p1.shape[1]
    d = (r - np.matmul(p1, np.transpose(p2))) // 2
    return d


#@timer
def calc_hamming_rank(b1, b2, force_slow=False):
    """Return rank of pairs. Takes vector of hashes b1 and b2 and returns correspondence rank of b1 to b2
    """
    if has_cython and b1.shape[1] < 33 and not force_slow:
        dist_h = _hamming.calc_hamming_dist64(b2, b1)
        return _hamming.sort(dist_h)
    elif has_cython and b1.shape[1] < 65 and not force_slow:
        dist_h = _hamming.calc_hamming_dist64(b2, b1)
        return _hamming.sort(dist_h)
    else:
        print("Warning. Using slow \"calc_hamming_dist\"")
        dist_h = calc_hamming_dist(b2, b1)
        return np.argsort(dist_h, 1, kind='mergesort')


# For testing
if __name__ == '__main__':
    b1 = np.random.rand(10, 48).astype(np.float32) - 0.5
    b2 = np.random.rand(20, 48).astype(np.float32) - 0.5
    d1_ = _hamming.calc_hamming_dist64(b2, b1)
    d2_ = calc_hamming_dist(b2, b1)
    print("Passed!" if (d1_ == d2_).all() else "Failed!")

    d1 = calc_hamming_rank(b1, b2)
    d2 = calc_hamming_rank(b1, b2, force_slow=True)
    print(d1)
    print(d2)

    print("Passed!" if (d1 == d2).all() else "Failed!")

    b1 = np.random.rand(500, 48).astype(np.float32) - 0.5
    b2 = np.random.rand(700, 48).astype(np.float32) - 0.5

    d1 = calc_hamming_rank(b1, b2)
    d2 = calc_hamming_rank(b1, b2, force_slow=True)

    print("Passed!" if (d1 == d2).all() else "Failed!")
