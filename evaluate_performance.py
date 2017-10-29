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
from utils import cifar10_reader

def evaluate(l_dataset, hashes_database, l_test, hashes_test, force_slow=False):
    """Evaluate MAP. Hardcoded numbers for CIFAR10 case. 1000 images per category, i.e. in total 10000 images,
    are randomly sampled as quire images (selection happens at preparation step). The remaining images are used
    as database images.
    """
    labels_database = np.reshape(np.asarray(l_dataset), [-1, 1])
    labels_test = np.reshape(np.asarray(l_test), [-1, 1])

    map_train = compute_map(
        hashes_database[:40000],
        hashes_database[40000:],
        labels_database[:40000],
        labels_database[40000:], force_slow)
    print("Test on train " + str(map_train))

    map_test = compute_map(hashes_database, hashes_test, labels_database, labels_test, force_slow)
    print("Test on test " + str(map_test))
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

    test, train = evaluate(items_train, items_test, b_database, b_test)
    test2, train2 = evaluate(items_train, items_test, b_database, b_test, True)

    print("Passed!" if (test == test2) and(train == train2) else "Failed!")

if __name__ == '__main__':
    evaluate_offline()
