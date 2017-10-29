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
"""Data preparation script for CIFAR10"""

import pickle
import os
from random import shuffle

from utils import cifar10_reader


def main():
    """Main function, run it if you import this file"""
    # Read all data (train and test), we are going to split later
    r = cifar10_reader.Reader('data/cifar-10-batches-bin', train=True, test=True)

    categorized = {key: [] for key in range(10)}

    for (label, img) in r.items:
        categorized[label].append(img)

    items_train = []
    items_test = []

    # Splitting data into 1000 for test and 10000 for train per category
    for i in range(10):
        shuffle(categorized[i])
        items_test += [(i, x) for x in categorized[i][0:1000]]
        items_train += [(i, x) for x in categorized[i][1000:]]

    shuffle(items_train)
    shuffle(items_test)

    if not os.path.exists('temp'):
        os.makedirs('temp')

    output = open('temp/items_train.pkl', 'wb')
    pickle.dump(items_train, output)
    output.close()

    output = open('temp/items_test.pkl', 'wb')
    pickle.dump(items_test, output)
    output.close()


if __name__ == '__main__':
    main()
