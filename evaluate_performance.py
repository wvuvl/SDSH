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
import pickle
from mean_average_precision import compute_map
from mean_average_precision import compute_map_fast
from utils import cifar10_reader
import time

def evaluate(l_train, hashes_train, l_test, hashes_test, l_db, hashes_db, top_n = 0, and_mode=False, force_slow=False, testOnTrain=False,weighted_mode = False):
    """Evaluate MAP. Hardcoded numbers for CIFAR10 case. 1000 images per category, i.e. in total 10000 images,
    are randomly sampled as quire images (selection happens at preparation step). The remaining images are used
    as database images.
    """
    labels_database = np.reshape(np.asarray(l_db), [-1, 1])
    labels_train = np.reshape(np.asarray(l_train), [-1, 1])
    labels_test = np.reshape(np.asarray(l_test), [-1, 1])

    hashes_database = hashes_db.astype(np.float32)
    hashes_train = hashes_train.astype(np.float32)
    hashes_test = hashes_test.astype(np.float32)

    map_train = 0.0

    if testOnTrain:
        map_train, _ = compute_map(
           hashes_train[:-1000],
           hashes_train[-1000:],
           labels_train[:-1000],
           labels_train[-1000:], top_n=top_n, and_mode=and_mode, force_slow=force_slow)
        #print("Test on train " + str(map_train))

    #pretime = time.perf_counter()
    map_test, curve = compute_map(hashes_database, hashes_test, labels_database, labels_test, top_n=top_n, and_mode=and_mode, force_slow=and_mode,weighted_mode = weighted_mode)
    #posttime = time.perf_counter()
    #print('time taken {}'.format(posttime-pretime))
    #print("Test on test " + str(map_test))
    return map_train, map_test, curve

def map():
    """Read pickled test, train sets and pickled hashes and perform evaluation"""
    with open('H.pkl', 'rb') as pkl:
        H = pickle.load(pkl).astype(np.float32)
    with open('labels.pkl', 'rb') as pkl:
        labels = pickle.load(pkl)
    with open('H_test.pkl', 'rb') as pkl:
        H_test = pickle.load(pkl).astype(np.float32)
    with open('labels_test.pkl', 'rb') as pkl:
        labels_test = pickle.load(pkl)

    m_fast = compute_map_fast(H_test, H_test, labels_test, labels_test
    , and_mode = True)
    m1, curve1 = compute_map(H_test, H_test, labels_test, labels_test
    , and_mode = True)
    m2, curve2 = compute_map(H_test, H_test, labels_test, labels_test
    , and_mode = True
    ,force_slow=True)

    print(m_fast)
    print(m1)
    print(m2)

if __name__ == '__main__':
    map()
