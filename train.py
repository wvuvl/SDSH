#! python3
import logging
import os
import pickle
import sys
import time
from pprint import pformat

import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

import batch_provider
import constructor
import loss_functions
from evaluate_performance import evaluate
from gen_hashes import gen_hashes


class Train:
    def __init__(self):
        self.formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(logging.DEBUG)
        self.console_handler.setFormatter(self.formatter)
        self.logger = None
        self.directory = None
        self.l_dataset = None
        self.b_dataset = None
        self.l_test = None
        self.b_test = None
        self.and_mode = False
        self.top_n = 0

        log_main = logging.getLogger()
        log_main.setLevel(logging.INFO)
        fh = logging.FileHandler("main.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(self.formatter)
        log_main.addHandler(fh)
        log_main.addHandler(self.console_handler)

    def run(self, path, config):
        class Cfg:
            def __init__(self):
                self.batch_size = 0
                self.loss = None
                self.margin = 0
                self.hash_size = 0
                self.weight_decay_factor = 0
                self.number_of_epochs_per_decay = 0
                self.learning_rate_decay_factor = 0
                self.learning_rate = 0
                self.total_epoch_count = 0
                self.dataset = None
                self.top_n = 0

        cfg = Cfg()
        self.cfg = cfg

        for key in config:
            setattr(cfg, key, config[key])

        name = "{0}_h{1}_m{2}_l{3}_d{4}".format(cfg.loss, cfg.hash_size, cfg.margin, cfg.learning_rate, cfg.weight_decay_factor)

        if cfg.dataset is not None:
            name = cfg.dataset + "_" + name

        directory = os.path.join(path, name)
        self.directory = directory

        self.top_n = cfg.top_n

        logging.info("Starting {0}...".format(name))

        if not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.exists(os.path.join(directory, "Done.txt")):
            logging.info("\tWas already finished, skipping {0}".format(name))
            return

        logger = logging.getLogger(name)
        logger.handlers = []
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        self.logger = logger

        file_handler = logging.FileHandler(os.path.join(directory, name + ".log"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(self.formatter)

        logger.addHandler(file_handler)
        logger.addHandler(self.console_handler)
        
        config = tf.ConfigProto(device_count = {'GPU': 1})

        with tf.Graph().as_default(), tf.Session(config=config) as session:
            logger.info("\n{0}\n{1}\n{0}\n".format("-" * 80, name))
            logger.info("\nSettings:\n{0}".format(pformat(vars(cfg))))

            items_test = []
            items_train = []
            items_db = []
            self.and_mode = False

            if cfg.dataset is None:
                with open('temp/items_train.pkl', 'rb') as pkl:
                    items_train = pickle.load(pkl)
                with open('temp/items_test.pkl', 'rb') as pkl:
                    items_test = pickle.load(pkl)
            elif cfg.dataset == "cifar_reduced":
                with open('temp/items_train_cifar_reduced.pkl', 'rb') as pkl:
                    items_train = pickle.load(pkl)
                with open('temp/items_test_cifar_reduced.pkl', 'rb') as pkl:
                    items_test = pickle.load(pkl)
                with open('temp/items_db_cifar_reduced.pkl', 'rb') as pkl:
                    items_db = pickle.load(pkl)
            elif cfg.dataset == "nus2100.10500":
                with open('temp/items_train_nuswide_2100.10500.pkl', 'rb') as pkl:
                    items_train = pickle.load(pkl)
                with open('temp/items_test_nuswide_2100.10500.pkl', 'rb') as pkl:
                    items_test = pickle.load(pkl)
                with open('temp/items_db_nuswide_2100.10500.pkl', 'rb') as pkl:
                    items_db = pickle.load(pkl)
                self.and_mode = True
                self.top_n = 5000
            elif cfg.dataset == "nus5000.10000":
                with open('temp/items_train_nuswide_5000.10000.pkl', 'rb') as pkl:
                    items_train = pickle.load(pkl)
                with open('temp/items_test_nuswide_5000.10000.pkl', 'rb') as pkl:
                    items_test = pickle.load(pkl)
                with open('temp/items_db_nuswide_5000.10000.pkl', 'rb') as pkl:
                    items_db = pickle.load(pkl)
                self.and_mode = True
                self.top_n = 5000
            elif cfg.dataset == "nus2100._":
                with open('temp/items_train_nuswide_2100._.pkl', 'rb') as pkl:
                    items_train = pickle.load(pkl)
                with open('temp/items_test_nuswide_2100._.pkl', 'rb') as pkl:
                    items_test = pickle.load(pkl)
                self.and_mode = True

            if len(items_db) > 0:
                l = (len(items_db) // 100) * 100
                items_db = items_db[:l]

            l = (len(items_train) // 100) * 100

            items_train = items_train[:l]
            print(len(items_db))
            print(len(items_train))

            assert(len(items_db) % 100 == 0)
            assert(len(items_train) % 100 == 0)
            assert(len(items_test) % 100 == 0)

31
            num_examples_per_epoch_for_train = len(items_train)

            bp = batch_provider.BatchProvider(cfg.batch_size, items_train, cycled=True)

            num_batches_per_epoch = num_examples_per_epoch_for_train / cfg.batch_size
            decay_steps = int(num_batches_per_epoch * cfg.number_of_epochs_per_decay)

            logger.info('decay_steps: ' + str(decay_steps))

            loss = loss_functions.losses[cfg.loss]
            model = constructor.net(cfg.batch_size, cfg.hash_size, cfg.margin, cfg.weight_decay_factor, loss)

            tf.summary.scalar('weigh_decay', model.weight_decay)
            tf.summary.scalar('total_loss_plus_weigh_decay', model.loss)
            tf.summary.image('embedding', tf.reshape(model.output, [-1, 1, cfg.hash_size, 1]))

            global_step = tf.contrib.framework.get_or_create_global_step()

            lr_modulator = tf.placeholder(tf.float32, shape=(), name="init")

            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(cfg.learning_rate,
                                            global_step,
                                            decay_steps,
                                            cfg.learning_rate_decay_factor,
                                            staircase=True)

            lr *= lr_modulator

            tf.summary.scalar('learning_rate', lr)

            weights_fc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 "fc*")

            opt = tf.train.GradientDescentOptimizer(lr)

            pre_train_step = opt.minimize(model.loss, global_step=global_step, var_list=weights_fc)
            train_step = opt.minimize(model.loss, global_step=global_step)
            _start_time = time.time()
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(directory, flush_secs=10, graph=session.graph)

            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)

            lc = tf.train.latest_checkpoint(directory)

            projector_config = projector.ProjectorConfig()
            embedding_conf = projector_config.embeddings.add()
            embedding_conf.tensor_name = 'embedding'
            embedding_conf.metadata_path = "metadata.tsv"
            projector.visualize_embeddings(writer, projector_config)

            start_step = 0

            if lc is not None:
                saver.restore(session, lc)
                start_step = session.run(global_step)

            batches = bp.get_batches()

            for i in range(start_step, int(cfg.total_epoch_count * num_batches_per_epoch)):
                feed_dict = next(batches)

                #if i < 2000:
                #    step = pre_train_step
                #else:
                #    step = train_step
                step = train_step

                labels = feed_dict["labels"]

                labels = np.asarray(labels, np.uint32)

                lr_mod = 1.0

                if self.and_mode:
                    mask = np.equal(np.reshape(labels, [cfg.batch_size, 1]), np.reshape(labels, [1, cfg.batch_size]))
                else:
                    mask = np.bitwise_and(np.reshape(labels, [cfg.batch_size, 1]),
                                          np.reshape(labels, [1, cfg.batch_size])).astype(dtype=np.bool)

                summary, _, _ = session.run(
                    [merged, model.assignment, step],
                    {
                        model.t_images: feed_dict["images"],
                        model.t_labels: feed_dict["labels"],
                        model.t_boolmask: mask,
                        lr_modulator: lr_mod
                    })

                writer.add_summary(summary, i)

                current_time = time.time()
                duration = current_time - _start_time
                _start_time = current_time

                examples_per_sec = cfg.batch_size / duration
                sec_per_batch = float(duration)

                format_str = ('step %d, (%.1f examples/sec; %.3f '
                              'sec/batch)')

                logger.debug(format_str % (i, examples_per_sec, sec_per_batch))

                if (i % 2000 == 0) and i != 0:
                    self.TestAndSaveCheckpoint(model, session, items_train, items_test, items_db, cfg.hash_size,
                                               directory, embedding_conf, saver, global_step, feed_dict)

            self.TestAndSaveCheckpoint(model, session, items_train, items_test, items_db, cfg.hash_size,
                                       directory, embedding_conf, saver, global_step)


        self.Rotation(cfg.hash_size, directory)

        with open(os.path.join(directory, "Done.txt"), "a") as file:
            file.write("\n")


    def TestAndSaveCheckpoint(self, model, session, items_train, items_test, items_db, hash_size,
                              directory, embedding_conf, saver, global_step, feed_dict=None):
        saver.save(session, os.path.join(directory, "checkpoint"), global_step)

        if feed_dict is not None:
            file = open(os.path.join(directory, embedding_conf.metadata_path), "w")
            for l in feed_dict["labels"]:
                file.write(str(l[0]) + "\n")
            file.close()

        self.logger.info("Start generating hashes")

        self.l_train, self.b_train = gen_hashes(model.t_images, model.t_labels,
                                       model.output, session, items_train, hash_size)

        self.l_test, self.b_test = gen_hashes(model.t_images, model.t_labels,
                                       model.output, session, items_test, hash_size, 1)

        if len(items_db) > 0:
            self.l_db, self.b_db = gen_hashes(model.t_images, model.t_labels,
                                       model.output, session, items_db, hash_size)
        else:
            self.l_db, self.b_db = self.l_train, self.b_train

        self.logger.info("Finished generating hashes")
        self.logger.info("Starting evaluation")

        map_train, map_test = evaluate(self.l_train, self.b_train, self.l_test, self.b_test, self.l_db, self.b_db, top_n=self.top_n, and_mode=self.and_mode)

        with open(os.path.join(directory, "results.txt"), "a") as file:
            file.write(str(map_train) + "\t" + str(map_test) + "\n")
        self.logger.info("Test on train: {0}, Test on test: {1}".format(map_train, map_test))

    def Rotation(self, hash_size, directory):
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = len(self.b_train)
        NUM_EPOCHS_PER_DECAY = 10.0
        LEARNING_RATE_DECAY_FACTOR = 0.5

        CLAMP = 0.03
        INITIAL_LEARNING_RATE = 0.07
        TOTAL_EPOCHS_COUNT = 30

        self.logger.info("Rotation")

        with tf.Graph().as_default(), tf.Session() as session:

            input_hash = tf.placeholder(tf.float32, [None, hash_size])
            input_label = tf.placeholder(tf.int32, [None, 1])

            r = tf.Variable(tf.eye(hash_size), dtype=tf.float32)

            r_tr = tf.matrix_band_part(r, 0, -1)
            r_tr *= tf.ones((hash_size, hash_size)) - tf.eye(hash_size)
            r_tr += -tf.transpose(r_tr)

            r_tr2 = tf.matmul(r_tr, r_tr)
            r_tr3 = tf.matmul(r_tr2, r_tr)
            r_tr4 = tf.matmul(r_tr3, r_tr)
            r_tr5 = tf.matmul(r_tr4, r_tr)
            r_tr6 = tf.matmul(r_tr5, r_tr)
            r_tr7 = tf.matmul(r_tr6, r_tr)
            r_tr8 = tf.matmul(r_tr7, r_tr)
            r_tr9 = tf.matmul(r_tr8, r_tr)
            r_tr10 = tf.matmul(r_tr9, r_tr)
            r_tr11 = tf.matmul(r_tr10, r_tr)
            r_tr12 = tf.matmul(r_tr11, r_tr)
            r_tr13 = tf.matmul(r_tr12, r_tr)
            rot = \
                tf.eye(hash_size) \
                + r_tr \
                + 1.0 / 2.0 * r_tr2\
                + 1.0 / 6.0 * r_tr3\
                + 1.0 / 24.0 * r_tr4\
                + 1.0 / 120.0 * r_tr5\
                + 1.0 / 720.0 * r_tr6\
                + 1.0 / 5040.0 * r_tr7\
                + 1.0 / 40320.0 * r_tr8\
                + 1.0 / 362880.0 * r_tr9\
                + 1.0 / 3628800.0 * r_tr10\
                + 1.0 / 39916800.0 * r_tr11\
                + 1.0 / 479001600.0 * r_tr12\
                + 1.0 / 6227020800.0 * r_tr13\


            b_r = tf.matmul(input_hash, rot)
            #b_r = tf.nn.l2_normalize(b_r, 1)

            batch_size = 1000

            t_boolmask = tf.placeholder(tf.bool, [batch_size, batch_size])

            positive_pairs = tf.cast(t_boolmask, tf.float32)
            negative_pairs = tf.cast(tf.logical_not(t_boolmask), tf.float32)

            mul = tf.multiply(tf.reshape(b_r, [1, batch_size, hash_size]), tf.reshape(b_r, [batch_size, 1, hash_size]))

            mul = tf.maximum(mul, -CLAMP)
            mul = tf.minimum(mul, CLAMP)

            negative = tf.reshape(negative_pairs, [batch_size, batch_size, 1]) * mul / CLAMP
            positive = tf.reshape(positive_pairs, [batch_size, batch_size, 1]) * mul / CLAMP

            reg = tf.reduce_mean(tf.square(tf.matmul(r, r, transpose_b=True) - tf.eye(hash_size)))

            loss = tf.reduce_mean(negative) - tf.reduce_mean(positive)# + reg

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('reg', reg)

            num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
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

            opt = tf.train.GradientDescentOptimizer(lr)
            train_step = opt.minimize(loss, global_step=global_step)

            _start_time = time.time()

            merged = tf.summary.merge_all()

            writer = tf.summary.FileWriter("F:\\tmp\\train2", flush_secs=10, graph=session.graph)

            session.run(tf.global_variables_initializer())

            k = 0
            for i in range(int(TOTAL_EPOCHS_COUNT * num_batches_per_epoch)):
                labels = self.l_train[k * batch_size:k * batch_size + batch_size]
                if self.and_mode:
                    mask = np.equal(np.reshape(labels, [batch_size, 1]), np.reshape(labels, [1, batch_size]))
                else:
                    mask = np.bitwise_and(np.reshape(labels, [batch_size, 1]),
                                          np.reshape(labels, [1, batch_size])).astype(dtype=np.bool)

                summary, _ = session.run([merged, train_step],
                                         {
                                             input_hash:  self.b_train[k * batch_size:k * batch_size + batch_size],
                                             input_label:  self.l_train[k * batch_size:k * batch_size + batch_size],
                                             t_boolmask: mask,
                                         })
                writer.add_summary(summary, i)

                k +=1
                if (k+1) * batch_size >= len(self.b_train):
                    k = 0

                current_time = time.time()
                duration = current_time - _start_time + 0.00001
                _start_time = current_time

                examples_per_sec = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (time.time(), i, examples_per_sec, sec_per_batch))

            tf_b_train = tf.constant(self.b_train, dtype=tf.float32)
            tf_b_db = tf.constant(self.b_db, dtype=tf.float32)
            tf_b_test = tf.constant(self.b_test, dtype=tf.float32)
            b_dataset_r = session.run(tf.matmul(tf_b_db, rot), {})
            b_train_r = session.run(tf.matmul(tf_b_train, rot), {})
            b_test_r = session.run(tf.matmul(tf_b_test, rot), {})

        self.logger.info("Finished learning rotation")
        self.logger.info("Starting evaluation")

        map_train, map_test = evaluate(self.l_train, b_train_r, self.l_test, b_test_r, self.l_db, b_dataset_r, top_n=self.top_n)

        with open(os.path.join(directory, "results.txt"), "a") as file:
            file.write("Rotation: " + str(map_train) + "\t" + str(map_test) + "\n")
        self.logger.info("Test on train: {0}, Test on test: {1}".format(map_train, map_test))
