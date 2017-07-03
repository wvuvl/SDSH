import numpy as np
from calcHammingRank import calc_hamming_rank
from calcMAP import calc_map


def return_map(b_train, b_test, s):
    order_h = calc_hamming_rank(b_train, b_test)
    return calc_map(order_h, np.transpose(s))
