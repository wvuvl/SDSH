#! python3
import tensorflow as tf
import numpy as np
import facehashinput
from datetime import datetime
import pickle
import time
from facehash_net import vgg_arg_scope
from facehash_net import vgg_19
from facehash_net import vgg_f
from facehash_net import loss

slim = tf.contrib.slim

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EPOCHS_PER_DECAY = 20.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0005      # Initial learning rate.
TOTAL_EPOCHS_COUNT = 200
HASH_SIZE = 24
BATCH_SIZE = 180

if __name__ == '__main__':
    sess = tf.Session()
    with open('items_train.pkl', 'rb') as pkl:
        items = pickle.load(pkl)

    bp = facehashinput.BatchProvider(BATCH_SIZE, items, cycled=True)

    t_images, t_labels = bp.inputs()

    sess = tf.Session()
    #with slim.arg_scope(vgg_arg_scope()):
    #    outputs, end_points = vgg_19(t_images - tf.reshape([123.68, 116.779, 103.939], [1, 1, 1, 3]))

    model = vgg_f('imagenet-vgg-f.mat', t_images)

    outputs = model.net['fc8']

    l = loss(outputs, t_labels, HASH_SIZE, BATCH_SIZE)

    #lc = "vgg_19.ckpt"
    #saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("vgg_19/fc8"))], keep_checkpoint_every_n_hours=1.0)
    #saver.restore(sess, lc)

    tf.summary.image('embedding', tf.reshape(outputs, [-1, 1, HASH_SIZE, 1]))

    tf.summary.scalar('loss', l)

    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
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

    writer = tf.summary.FileWriter("F:\\tmp\\vgg", flush_secs=10, graph=sess.graph)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)

    lc = tf.train.latest_checkpoint("F:\\tmp\\vgg\\")

    if lc is not None:
        saver.restore(sess, lc)

    for i in range(int(TOTAL_EPOCHS_COUNT * num_batches_per_epoch)):
        feed_dict = bp.get_batch()
        if i%1000 == 0:
            saver.save(sess, "F:\\tmp\\vgg\\checkpoint", global_step)

        summary, _ = sess.run([merged, train_step], feed_dict)
        writer.add_summary(summary, i)

        current_time = time.time()
        duration = current_time - _start_time
        _start_time = current_time

        examples_per_sec = BATCH_SIZE / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, (%.1f examples/sec; %.3f '
                          'sec/batch)')
        print(format_str % (datetime.now(), i, examples_per_sec, sec_per_batch))

