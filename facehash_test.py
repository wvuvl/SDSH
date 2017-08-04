#! python3
import numpy as np
import facehashinput
import pickle
from compute_S import compute_s
from return_map import return_map
from datetime import datetime
from facehash_net import vgg_f
import time

HASH_SIZE = 24


def test(items_train, items_test, b_dataset, b_test):
    r_train = facehashinput.Reader('', items_train)
    r_test = facehashinput.Reader('', items_test)

    dataset_l = np.reshape(np.asarray(r_train.get_labels()), [-1, 1])
    test_l = np.reshape(np.asarray(r_test.get_labels()), [-1, 1])

    s = compute_s(dataset_l[:40000], dataset_l[40000:])
    map_train = return_map(b_dataset[:40000], b_dataset[40000:], s)
    print("Test on train" + str(map_train))

    s = compute_s(dataset_l, test_l)
    map_test = return_map(b_dataset, b_test, s)
    print("Test on test" + str(map_test))
    return map_train, map_test

if __name__ == '__main__':
    with open('items_train.pkl', 'rb') as pkl:
        items_train = pickle.load(pkl)
    with open('items_test.pkl', 'rb') as pkl:
        items_test = pickle.load(pkl)
    with open('b_dataset.pkl', 'rb') as pkl:
        b_dataset = pickle.load(pkl)
    with open('b_test.pkl', 'rb') as pkl:
        b_test = pickle.load(pkl)

    test(items_train, items_test, b_dataset, b_test)
