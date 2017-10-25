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
from compute_S import compute_s
from return_map import return_map
from utils import cifar10_reader


def evaluate(items_train, items_test, b_database, b_test):
    """Evaluate MAP. Hardcoded numbers for CIFAR10 case. 1000 images per category, i.e. in total 10000 images,
    are randomly sampled as quire images (selection happens at preparation step). The remaining images are used
    as database images.
    """
    r_train = cifar10_reader.Reader('', items_train)
    r_test = cifar10_reader.Reader('', items_test)

    database_l = np.reshape(np.asarray(r_train.get_labels()), [-1, 1])
    test_l = np.reshape(np.asarray(r_test.get_labels()), [-1, 1])

    s = compute_s(database_l[:40000], database_l[40000:])
    map_train = return_map(b_database[:40000], b_database[40000:], s)
    print("Test on train" + str(map_train))

    s = compute_s(database_l, test_l)
    map_test = return_map(b_database, b_test, s)
    print("Test on test" + str(map_test))
    return map_train, map_test


def evaluate_offline():
    """Read pickled test, train sets and pickled hashes and perform evaluation"""
    with open('temp/items_train.pkl', 'rb') as pkl:
        items_train = pickle.load(pkl)
    with open('temp/items_test.pkl', 'rb') as pkl:
        items_test = pickle.load(pkl)
    with open('temp/b_database.pkl', 'rb') as pkl:
        b_database = pickle.load(pkl)
    with open('temp/b_test.pkl', 'rb') as pkl:
        b_test = pickle.load(pkl)

    evaluate(items_train, items_test, b_database, b_test)

if __name__ == '__main__':
    evaluate_offline()
