import numpy as np


def compute_s(train_l, test_l):
    d = np.tile(train_l, (1, test_l.shape[0])) - np.tile(np.transpose(test_l), (train_l.shape[0], 1))
    return np.equal(d, 0)
