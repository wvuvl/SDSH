import numpy as np
from compute_B import compute_b
from compute_S import compute_s
from return_map import return_map


def test(net, dataset_l, test_l, data_set, test_data):
    s = compute_s(dataset_l, test_l)
    (b_dataset, b_test) = compute_b(data_set, test_data, net)
    return return_map(b_dataset, b_test, s)
