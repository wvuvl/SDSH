import numpy as np


# compute the hamming distance between every pair of data points represented in each row of B1 and B2
def calc_hamming_dist(b1, b2):
    p1 = np.sign(b1)
    p2 = np.sign(b2)

    r = p1.shape[1]
    d = np.round(r - np.matmul(p1, np.transpose(p2)) / 2.0).astype(int)
    return d
