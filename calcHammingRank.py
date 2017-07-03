import numpy as np
from calcHammingDist import calc_hamming_dist


def calc_hamming_rank(b1, b2):
    dist_h = calc_hamming_dist(b2, b1)
    return np.argsort(dist_h, 1)
