#! python3
import numpy as np
import facehashinput
import pickle
from compute_S import compute_s
from return_map import return_map
from datetime import datetime
from matconvnet2tf import vgg_f
import time
import tensorflow as tf
import math
from loss_functions import loss2

HASH_SIZE = 24

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EPOCHS_PER_DECAY = 1000.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01  # Initial learning rate.
TOTAL_EPOCHS_COUNT = 2000

if __name__ == '__main__':
    sess = tf.Session()

    with open('items_train.pkl', 'rb') as pkl:
        items_train = pickle.load(pkl)
    with open('items_test.pkl', 'rb') as pkl:
        items_test = pickle.load(pkl)

    with open('b_dataset.pkl', 'rb') as pkl:
        b_dataset = pickle.load(pkl)

    with open('b_test.pkl', 'rb') as pkl:
        b_test = pickle.load(pkl)

    r_train = facehashinput.Reader('', items_train)
    train_l = np.reshape(np.asarray(r_train.get_labels()), [-1, 1])

    r = tf.Variable(tf.eye(HASH_SIZE), dtype=tf.float32)
    b = tf.constant(b_dataset, dtype=tf.float32)

    b = tf.nn.l2_normalize(b, 1)

    b = tf.matmul(tf.reshape(r, [1, HASH_SIZE, HASH_SIZE]) * tf.ones([b.shape[0], 1, 1]), tf.reshape(b, [-1, HASH_SIZE, 1]))

    b_t = tf.constant(b_test, dtype=tf.float32)
    b_t = tf.nn.l2_normalize(b_t, 1)
    b_t = tf.matmul(tf.reshape(r, [1, HASH_SIZE, HASH_SIZE]) * tf.ones([b_t.shape[0], 1, 1]), tf.reshape(b_t, [-1, HASH_SIZE, 1]))

    #l = tf.reduce_mean(tf.square(tf.abs(b) - 1.0)) + tf.reduce_mean(tf.square(tf.matmul(r, r, transpose_b=True) - tf.eye(HASH_SIZE)))
    b_e = 2.0 / (1.0 + tf.exp(-7.0 * b)) - 1.0
    l = tf.reduce_mean(tf.square(tf.reduce_mean(b_e, 0))) + 100.0 * tf.reduce_mean(tf.square(tf.matmul(r, r, transpose_b=True) - tf.eye(HASH_SIZE)))
    #l = loss2(b, 24, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN) + 100.0 * tf.reduce_mean(tf.square(tf.matmul(r, r, transpose_b=True) - tf.eye(HASH_SIZE)))

    tf.summary.scalar('loss', l)

    num_batches_per_epoch = 1
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    print('decay_steps: ' + str(decay_steps))

    global_step = tf.contrib.framework.get_or_create_global_step()

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    opt = tf.train.AdamOptimizer(lr)
    train_step = opt.minimize(l, global_step=global_step)

    _start_time = time.time()

    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter("F:\\tmp\\train2", flush_secs=10, graph=sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(int(TOTAL_EPOCHS_COUNT * num_batches_per_epoch)):
        summary, _ = sess.run([merged, train_step], {})
        writer.add_summary(summary, i)

        current_time = time.time()
        duration = current_time - _start_time + 0.00001
        _start_time = current_time

        examples_per_sec = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (datetime.now(), i, examples_per_sec, sec_per_batch))

    b_dataset_r = sess.run(tf.reshape(b, [-1, HASH_SIZE]), {})

    output = open('b_dataset_r.pkl', 'wb')
    pickle.dump(b_dataset_r, output)
    output.close()

    b_test_r = sess.run(tf.reshape(b_t, [-1, HASH_SIZE]), {})

    output = open('b_test_r.pkl', 'wb')
    pickle.dump(b_test_r, output)
    output.close()

def get_r(k, theta):
    return tf.eye(k.shape[1], batch_shape=k.shape[0]) + tf.sin(theta) * k + (1.0 - tf.cos(theta) * tf.matmul(k, k))
