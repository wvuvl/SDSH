#! python3
import numpy as np
import batch_provider
import pickle
from constructor import net
import tensorflow as tf

BATCH_SIZE = 100 # must be a divider of 10000 and 50000


def gen_hashes(t_images, t_labels, outputs, sess, items, hash_size, worker=16):
    bp = batch_provider.BatchProvider(BATCH_SIZE, items, worker=1, cycled=False)
	
    b = np.zeros([len(items), hash_size])
    l = np.zeros([len(items), 1], dtype=np.int32)
	
    batches = bp.get_batches()

    k = 0

    while True:
        feed_dict = next(batches)
        if feed_dict is None:
            break

        result = sess.run(outputs, {t_images: feed_dict["images"], t_labels: feed_dict["labels"]})

        b[k: k + BATCH_SIZE] = result
        l[k: k + BATCH_SIZE] = feed_dict["labels"]

        k += BATCH_SIZE

    print(len(b))
    print(k)
    assert(len(b) == k)
    assert(len(l) == k)

    return l, b
