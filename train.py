#! python3
import logging
import os
import pickle
import sys
import time
from pprint import pformat
from shutil import copyfile

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
        self.l_db = None
        self.b_db = None
        self.l_test = None
        self.b_test = None
        self.and_mode = False
        self.top_n = 0
        self.FAcc =0

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
                self.freeze = False

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

        # Structude
        # path_to_train | path_to_test | path_to_db(None if not applicable) | if true labels are compared by equality op, otherwise by and op. | top_n
        data_dict = {
            "cifar_full":        ['items_train.pkl',                    'items_test.pkl',                     None,                                     False,      0],
            "cifar_reduced":     ['items_train_cifar_reduced.pkl',      'items_test_cifar_reduced.pkl',       'items_db_cifar_reduced.pkl',             False,      0],
            "nus2100.10500":     ['items_train_nuswide_2100.10500.pkl', 'items_test_nuswide_2100.10500.pkl',  'items_db_nuswide_2100.10500.pkl',        True,    5000],
            "nus5000.10000":     ['items_train_nuswide_5000.10000.pkl', 'items_test_nuswide_5000.10000.pkl',  'items_db_nuswide_5000.10000.pkl',        True,    5000],
            "nus2100._":         ['items_train_nuswide_2100._.pkl',     'items_test_nuswide_2100._.pkl',      None,                                     True,   50000],
            "imagenet":          ['items_train_imagenet.pkl',           'items_test_imagenet.pkl',            'items_db_imagenet.pkl',                  False,   5000],
            "mnist":             ['mnist_train.pkl',                    'mnist_test.pkl',                     False,                                    False,      0],
        }

        if cfg.dataset is None:
            cfg.dataset = "cifar_full"

        with tf.Graph().as_default(), tf.Session(config=config) as session:
            logger.info("\n{0}\n{1}\n{0}\n".format("-" * 80, name))
            logger.info("\nSettings:\n{0}".format(pformat(vars(cfg))))

            items_db = []
            self.and_mode = data_dict[cfg.dataset][3]
            self.top_n = data_dict[cfg.dataset][4]

            ## Save dataset partitions
            copyfile(os.path.join('temp', data_dict[cfg.dataset][0]), os.path.join(path, data_dict[cfg.dataset][0]))
            copyfile(os.path.join('temp', data_dict[cfg.dataset][1]), os.path.join(path, data_dict[cfg.dataset][1]))
            if data_dict[cfg.dataset][2] is not None:
                copyfile(os.path.join('temp', data_dict[cfg.dataset][2]), os.path.join(path, data_dict[cfg.dataset][2]))

            with open('temp/' + data_dict[cfg.dataset][0], 'rb') as pkl:
                items_train = pickle.load(pkl)
            with open('temp/' + data_dict[cfg.dataset][1], 'rb') as pkl:
                items_test = pickle.load(pkl)

            if data_dict[cfg.dataset][2] is not None:
                with open('temp/' + data_dict[cfg.dataset][2], 'rb') as pkl:
                    items_db = pickle.load(pkl)

            # Should be divisible by 100
            # The reason is to keep testing procedure simple. For testing size of batch is 100
            # and in a such way, we do not have reminder.
            # Testing sets so far all divisible by 100
            # Just pad db and training sets to make them also divisible by 100

            def pad(array):
                if len(array) % 100 != 0:
                    padding = 100 - len(array) % 100
                    array += array[:padding]

            pad(items_db)
            pad(items_train)
            #Do not pad test. We want to keep everything fair, will fail if not divisible by 100 on assert below
            #pad(items_test)

            print('DB set size: %d' % len(items_db))
            print('Train set size: %d' % len(items_train))
            print('Test set size: %d' % len(items_test))

            assert(len(items_db) % 100 == 0)
            assert(len(items_train) % 100 == 0)
            assert(len(items_test) % 100 == 0)

            num_examples_per_epoch_for_train = len(items_train)

            bp = batch_provider.BatchProvider(cfg.batch_size, items_train, cycled=True, imagenet=cfg.dataset == "imagenet")

            num_batches_per_epoch = num_examples_per_epoch_for_train / cfg.batch_size
            decay_steps = int(num_batches_per_epoch * cfg.number_of_epochs_per_decay)

            logger.info('decay_steps: ' + str(decay_steps))

            loss = loss_functions.losses[cfg.loss]
            model = constructor.net(cfg.batch_size, cfg.hash_size, cfg.margin, cfg.weight_decay_factor, loss)

            tf.summary.scalar('weigh_decay', model.weight_decay)
            tf.summary.scalar('total_loss_plus_weigh_decay', model.loss)
            tf.summary.image('embedding', tf.reshape(model.output, [-1, 1, cfg.hash_size, 1]))

            global_step = tf.contrib.framework.get_or_create_global_step()

            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(cfg.learning_rate,
                                            global_step,
                                            decay_steps,
                                            cfg.learning_rate_decay_factor,
                                            staircase=True)

            tf.summary.scalar('learning_rate', lr)

            weights_fc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 "fc*")

            opt = tf.train.GradientDescentOptimizer(lr)

            fcn_train_step = opt.minimize(model.loss, global_step=global_step, var_list=weights_fc)
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

                if cfg.freeze and i < 500:
                    step = fcn_train_step
                else:
                    step = train_step

                labels = feed_dict["labels"]

                if self.and_mode:
                    labels = np.asarray(labels, np.object)
                else:
                    labels = np.asarray(labels, np.uint32)

                if self.and_mode:
                    mask = np.bitwise_and(np.reshape(labels, [cfg.batch_size, 1]),
                                          np.reshape(labels, [1, cfg.batch_size])).astype(dtype=np.bool)
                else:
                    mask = np.equal(np.reshape(labels, [cfg.batch_size, 1]), np.reshape(labels, [1, cfg.batch_size]))

                summary, _, _ = session.run(
                    [merged, model.assignment, step],
                    {
                        model.t_images: feed_dict["images"],
                        #model.t_labels: feed_dict["labels"],
                        model.t_boolmask: mask,
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

        self.Rotation(directory)

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

        longints = self.and_mode

        self.l_train, self.b_train = gen_hashes(model.t_images, model.t_labels,
                                       model.output, session, items_train, hash_size, longints=longints, imagenet=self.cfg.dataset == "imagenet")

        self.l_test, self.b_test = gen_hashes(model.t_images, model.t_labels,
                                       model.output, session, items_test, hash_size, 1, longints=longints, imagenet=self.cfg.dataset == "imagenet")

        if len(items_db) > 0:
            self.l_db, self.b_db = gen_hashes(model.t_images, model.t_labels,
                                       model.output, session, items_db, hash_size, longints=longints, imagenet=self.cfg.dataset == "imagenet")
        else:
            self.l_db, self.b_db = self.l_train, self.b_train

        self.logger.info("Finished generating hashes")

        map_train, map_test = self.eval(directory, self.l_train, self.b_train, self.l_test, self.b_test, self.l_db, self.b_db)

        self.FAcc = map_test

    def Rotation(self, directory):
        self.logger.info("Starting rotations")
        labels = self.l_train
        H = self.b_train

        size = labels.shape[0]

        if size > 25000:
            idx = np.random.randint(size, size=25000)
            size = 25000
            labels = labels[idx,:]
            H = H[idx,:]

        if self.and_mode:
            S = np.bitwise_and(np.reshape(labels, [size, 1]),
                               np.reshape(labels, [1, size])).astype(dtype=np.bool)
        else:
            S = np.equal(np.reshape(labels, [size, 1]), np.reshape(labels, [1, size]))

        R = np.eye(self.cfg.hash_size, self.cfg.hash_size, dtype=np.float32)
        for i in range(500):
            #update B
            B = np.matmul(S, np.sign(np.matmul(H, R)))

            #update R
            U, s, Vh = np.linalg.svd(np.matmul(B.T, H), full_matrices=False)
            R = np.matmul(Vh.T, U.T)

        b_train_r = np.matmul(self.b_train, R)
        b_test_r = np.matmul(self.b_test, R)
        b_db_r = np.matmul(self.b_db, R)
        self.logger.info("Finished rotations")

        self.eval(directory, self.l_train, b_train_r, self.l_test, b_test_r, self.l_db, b_db_r, True)
        return

    def eval(self, directory, l_train, b_train, l_test, b_test, l_db, b_db, rotations=False):
        self.logger.info("Starting evaluation")
        map_train, map_test, curve = evaluate(
              l_train
            , b_train
            , l_test
            , b_test
            , l_db
            , b_db
            , top_n=self.top_n
            , and_mode=self.and_mode
            , force_slow=self.and_mode)

        if rotations:
            report_string = "Test on train: {0}; Test on test: {1}".format(map_train, map_test)
        else:
            report_string = "Rotation: Test on train: {0}; Test on test: {1}".format(map_train, map_test)

        with open(os.path.join(directory, "results.txt"), "a") as file:
            file.write(report_string + "\n")
        self.logger.info(report_string)

        output = open(os.path.join(directory, "pr_curve.pkl"), 'wb')
        pickle.dump(curve, output)
        output.close()

        return map_train, map_test
