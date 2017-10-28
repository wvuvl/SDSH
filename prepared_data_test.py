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
"""Test pickled data. To insure that the prepared, pickled data is correct."""

import pickle
import batch_provider
import matplotlib.pyplot as plt


def main():
    """Load a batch of 10 items from test and train pickled data and display it"""
    with open('temp/items_train.pkl', 'rb') as pkl:
        items_train = pickle.load(pkl)
    with open('temp/items_test.pkl', 'rb') as pkl:
        items_test = pickle.load(pkl)

    bp_train = batch_provider.BatchProvider(40, items_train, cycled=True)
    bp_test = batch_provider.BatchProvider(40, items_test, cycled=True)

    it = bp_train.get_batches()
    b = next(it)
    for l in b["labels"]:
        print(l)
    for im in b["images"]:
        plt.imshow(im, interpolation='nearest')
        plt.show()

    it = bp_test.get_batches()
    b = next(it)
    for l in b["labels"]:
        print(l)
    for im in b["images"]:
        plt.imshow(im, interpolation='nearest')
        plt.show()

if __name__ == '__main__':
    main()
