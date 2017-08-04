#! python3
import tensorflow as tf
import numpy as np
import facehashinput
from datetime import datetime
import pickle
import time
import os
import sys
from facehash_net import vgg_f
from facehash_net import loss
from facehash_net import loss2
from facehash_net import loss_accv
from tensorflow.contrib.tensorboard.plugins import projector

from facehash_genHashes import GenHashes
from facehash_test import test

slim = tf.contrib.slim

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EPOCHS_PER_DECAY = 20.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 2.0 / 3.0  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.025      # Initial learning rate.
TOTAL_EPOCHS_COUNT = 100
HASH_SIZE = 24
BATCH_SIZE = 150
WEIGHT_DECAY_FACTOR = 5.0e-5

def main(LOG_FOLDER = "F:/tmp/vgg", MARGIN=1.0):
    #WEIGHT_DECAY_FACTOR = float(WEIGHT_DECAY_FACTOR)
    MARGIN = float(MARGIN)
    print("Using weightdecay: " + str(WEIGHT_DECAY_FACTOR))
    print("Using MARGIN: " + str(MARGIN))

    with open('items_train.pkl', 'rb') as pkl:
        items_train = pickle.load(pkl)
    with open('items_test.pkl', 'rb') as pkl:
        items_test = pickle.load(pkl)

    bp = facehashinput.BatchProvider(BATCH_SIZE, items_train, cycled=True)

    t_images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    t_labels = tf.placeholder(tf.int32, [None, 1])

    sess = tf.Session()

    model = vgg_f('imagenet-vgg-f.mat', t_images)

    outputs = model.net['fc8']

    embedding_norm = tf.nn.l2_normalize(outputs, 1)

    embedding_var = tf.Variable(tf.zeros((BATCH_SIZE, HASH_SIZE), dtype=tf.float32), trainable=False, name='embedding', dtype='float32')
    assignment = tf.assign(embedding_var, embedding_norm)

    weigh_decay = model.weight_decay * WEIGHT_DECAY_FACTOR
    l = loss(outputs, t_labels, HASH_SIZE, BATCH_SIZE, MARGIN) + weigh_decay

    tf.summary.scalar('weigh_decay', weigh_decay)
    tf.summary.scalar('total_loss_plus_weigh_decay', l)

    tf.summary.image('embedding', tf.reshape(outputs, [-1, 1, HASH_SIZE, 1]))

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

    #opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.GradientDescentOptimizer(2e-2)
    train_step = opt.minimize(l, global_step=global_step)

    _start_time = time.time()

    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter(LOG_FOLDER, flush_secs=10, graph=sess.graph)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)

    lc = tf.train.latest_checkpoint(LOG_FOLDER)

    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = 'embedding'
    embedding_conf.metadata_path = os.path.join(LOG_FOLDER, "metadata.tsv")

    projector.visualize_embeddings(writer, config)

    if lc is not None:
        saver.restore(sess, lc)

    feed_dict = {}
        
    for i in range(int(TOTAL_EPOCHS_COUNT * num_batches_per_epoch)):
        feed_dict = bp.get_batch()
        summary, _, _ = sess.run([merged, assignment, train_step], {t_images: feed_dict["images"], t_labels: feed_dict["labels"]})
        writer.add_summary(summary, i)

        current_time = time.time()
        duration = current_time - _start_time
        _start_time = current_time

        examples_per_sec = BATCH_SIZE / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, (%.1f examples/sec; %.3f '
                          'sec/batch)')
        if (i%2000 == 0) and i != 0:
            TestAndSaveCheckpoint(t_images, t_labels, outputs, sess, items_train, items_test, LOG_FOLDER, embedding_conf, saver, global_step, feed_dict)

        print(format_str % (datetime.now(), i, examples_per_sec, sec_per_batch))
    
    TestAndSaveCheckpoint(t_images, t_labels, outputs, sess, items_train, items_test, LOG_FOLDER, embedding_conf, saver, global_step, feed_dict)


def TestAndSaveCheckpoint(t_images, t_labels, outputs, sess, items_train, items_test, LOG_FOLDER, embedding_conf, saver, global_step, feed_dict=None):
    saver.save(sess, os.path.join(LOG_FOLDER, "checkpoint"), global_step)

    if feed_dict is not None:
        file = open(embedding_conf.metadata_path, "w")
        for l in feed_dict["labels"]:
            file.write(str(l[0]) + "\n")
        file.close()

    #b_dataset, b_test = GenHashes(t_images, t_labels, outputs, sess, items_train, items_test)

    #map_train, map_test = test(items_train, items_test, b_dataset, b_test)

    #with open(os.path.join(LOG_FOLDER, "log.txt"), "a") as file:
    #    file.write(str(map_train) + "\t" + str(map_test) + "\n")

if __name__ == '__main__':
    if len(sys.argv) == 3:
        print("Path: " + str(sys.argv[1]))
        print("MARGIN: " + str(sys.argv[2]))
        main(sys.argv[1], sys.argv[2])
    else:
        main()
