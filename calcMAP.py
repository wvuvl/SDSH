import numpy as np


# compute mean average precision (MAP)
def calc_map (order_h, neighbor):
    (Q, N) = neighbor.shape
    pos = range(1, N + 1)
    map = 0
    num_succ = 0
    for i in range(Q):
        ngb = neighbor[i, order_h[i, :]]
        n_rel = np.sum(ngb)
        if n_rel > 0:
            prec = np.cumsum(ngb) / pos
            ap = np.mean(prec[ngb])
            map += ap
            num_succ += 1

    map /= num_succ
    num_succ /= Q
    return map, num_succ
