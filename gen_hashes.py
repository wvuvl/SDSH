#! python3
import numpy as np
import batch_provider
import pickle
from constructor import net
import tensorflow as tf

BATCH_SIZE = 100 # must be a divider of 10000 and 50000


def gen_hashes(t_images, t_labels, outputs, sess, items_train, items_test, hash_size):
    bp_train = batch_provider.BatchProvider(BATCH_SIZE, items_train, cycled=False)
    bp_test = batch_provider.BatchProvider(BATCH_SIZE, items_test, cycled=False)

    b_dataset = np.zeros([len(items_train), hash_size])
    l_dataset = np.zeros([len(items_train), 1], dtype=np.int32)

    batches_train = bp_train.get_batches()

    k = 0

    while True:
        feed_dict = next(batches_train)
        if feed_dict is None:
            break

        result = sess.run(outputs, {t_images: feed_dict["images"], t_labels: feed_dict["labels"]})

        b_dataset[k: k + BATCH_SIZE] = result
        l_dataset[k: k + BATCH_SIZE] = feed_dict["labels"]

        k += BATCH_SIZE

    assert(len(b_dataset) == k)
    assert(len(l_dataset) == k)

    b_test = np.empty([len(items_test), hash_size])
    l_test = np.empty([len(items_test), 1], dtype=np.int32)

    batches_test = bp_test.get_batches()

    k = 0

    while True:
        feed_dict = next(batches_test)
        if feed_dict is None:
            break

        result = sess.run(outputs, {t_images: feed_dict["images"], t_labels: feed_dict["labels"]})

        b_test[k: k + BATCH_SIZE] = result
        l_test[k: k + BATCH_SIZE] = feed_dict["labels"]

        k += BATCH_SIZE

    assert (len(b_test) == k)
    assert (len(l_test) == k)

    return l_dataset, b_dataset, l_test, b_test


def test():
    sess = tf.Session()

    model = net(150, 24)

    output = model.output

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    lc = tf.train.latest_checkpoint("F:\\tmp\\vgg\\")
    saver.restore(sess, lc)

    with open('items_train.pkl', 'rb') as pkl:
        items_train = pickle.load(pkl)
    with open('items_test.pkl', 'rb') as pkl:
        items_test = pickle.load(pkl)

        l_dataset, b_dataset, l_test, b_test = gen_hashes(model.t_images, model.t_labels, output, sess, items_train, items_test, 24)

    output = open('b_dataset.pkl', 'wb')
    pickle.dump(l_dataset, output)
    pickle.dump(b_dataset, output)
    output.close()
    output = open('b_test.pkl', 'wb')
    pickle.dump(l_test, output)
    pickle.dump(b_test, output)
    output.close()

if __name__ == '__main__':
    test()
