#! python3
import numpy as np
import batch_provider
import pickle
from constructor import net
import tensorflow as tf

BATCH_SIZE = 100 # must be a divider of 10000 and 50000


def gen_hashes(t_images, prob, outputs, sess, items, batch_provider_constructor, longints=False):
    bp = batch_provider_constructor(items, False, BATCH_SIZE)

    if len(outputs.shape) != 2:
        shape = outputs.get_shape().as_list()[1:]
        outputs = tf.reshape(outputs, [-1, shape[0] * shape[1] * shape[2]])

    output_size = outputs.shape[1]

    b = np.zeros([len(items), output_size])

    if longints:
        l = np.zeros([len(items), 1], dtype=np.object)
    else:
        l = np.zeros([len(items), 1], dtype=np.uint32)

    batches = bp.get_batches()

    k = 0

    while True:
        feed_dict = next(batches)
        if feed_dict is None:
            break

        result = sess.run(outputs, {t_images: feed_dict["images"],
                                    prob: 1.0,})

        b[k: k + BATCH_SIZE] = result
        l[k: k + BATCH_SIZE] = feed_dict["labels"]

        k += BATCH_SIZE

    if (len(b) != k) or (len(l) != k):
        print(len(b))
        print(len(l))
        print(k)
        assert(len(b) == k)
        assert(len(l) == k)

    return l, b
