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
from mean_average_precision import compute_map
from mean_average_precision import compute_map_fast
from utils.random_rotation import random_rotation
from random import random
import threading


def GetBaseRotation(alpha, size):
    alphas = np.sin(alpha)
    alphac = np.cos(alpha)
    flat_rotation = np.array([[alphac, -alphas], [alphas, alphac]])
    I = np.eye(size)
    I[0:2, 0:2] = flat_rotation
    return I.astype(np.float32)


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

        samples_comparison_method = {
            "equality":0,
            "and":1,
            "weighted":2,
        }

        # Structude
        # path_to_train | path_to_test | path_to_db(None if not applicable) | sample comparison method | top_n
        data_dict = {
            "cifar_full":        ['items_train.pkl',                    'items_test.pkl',                     None,                                     "equality",  0],
            "cifar_reduced":     ['items_train_cifar_reduced.pkl',      'items_test_cifar_reduced.pkl',       'items_db_cifar_reduced.pkl',             "equality",  0],
            "nus2100.10500":     ['items_train_nuswide_2100.10500.pkl', 'items_test_nuswide_2100.10500.pkl',  'items_db_nuswide_2100.10500.pkl',        "and",       5000],
            "nus5000.10000":     ['items_train_nuswide_5000.10000.pkl', 'items_test_nuswide_5000.10000.pkl',  'items_db_nuswide_5000.10000.pkl',        "and",       5000],
            "nus2100._":         ['items_train_nuswide_2100._.pkl',     'items_test_nuswide_2100._.pkl',      None,                                     "and",       50000],
            "imagenet":          ['items_train_imagenet.pkl',           'items_test_imagenet.pkl',            'items_db_imagenet.pkl',                  "equality",  5000],
            "mnist":             ['mnist_train.pkl',                    'mnist_test.pkl',                     None,                                     "equality",   0],
            "mirflickr":         ['mirflickr25train.pkl',               'mirflickr25test.pkl',                None,                                     "weighted",  0],
        }

        if cfg.dataset is None:
            cfg.dataset = "cifar_full"

        with tf.Graph().as_default(), tf.Session(config=config) as session:
            logger.info("\n{0}\n{1}\n{0}\n".format("-" * 80, name))
            logger.info("\nSettings:\n{0}".format(pformat(vars(cfg))))

            items_db = []
            self.and_mode = samples_comparison_method[data_dict[cfg.dataset][3]]
            self.top_n = data_dict[cfg.dataset][4]

            ## Save dataset partitions

            copyfile(os.path.join('temp', data_dict[cfg.dataset][0]), os.path.join(path, data_dict[cfg.dataset][0]))
            copyfile(os.path.join('temp', data_dict[cfg.dataset][1]), os.path.join(path, data_dict[cfg.dataset][1]))
            if data_dict[cfg.dataset][2] is not None:
                copyfile(os.path.join('temp', data_dict[cfg.dataset][2]), os.path.join(path, data_dict[cfg.dataset][2]))

            with open('temp/' + data_dict[cfg.dataset][0], 'rb') as pkl:
                print(data_dict[cfg.dataset][0])
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
            lmdb_file = './data/mirf' if cfg.dataset == 'mirflickr' else None
            bp = batch_provider.BatchProvider(cfg.batch_size, items_train, cycled=True, imagenet=cfg.dataset == "imagenet",lmdb_file=lmdb_file)

            num_batches_per_epoch = num_examples_per_epoch_for_train / cfg.batch_size
            decay_steps = int(num_batches_per_epoch * cfg.number_of_epochs_per_decay)

            logger.info('decay_steps: ' + str(decay_steps))

            loss = loss_functions.losses[cfg.loss]
            model = constructor.net(cfg.batch_size, cfg.hash_size, cfg.margin, cfg.weight_decay_factor, loss)

            tf.summary.scalar('weigh_decay', model.weight_decay)
            tf.summary.scalar('total_loss', model.loss)
            model.loss += model.weight_decay
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
            #opt = tf.train.MomentumOptimizer(lr, momentum=0.9)

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

                if cfg.freeze:# and i < 500:
                    step = fcn_train_step
                else:
                    step = train_step

                labels = feed_dict["labels"]

                if self.and_mode == 1:
                    labels = np.asarray(labels, np.object)
                else:
                    labels = np.asarray(labels, np.uint32)

                if self.and_mode == 1 or self.and_mode == 2:
                    mask = np.bitwise_and(np.reshape(labels, [cfg.batch_size, 1]),
                                          np.reshape(labels, [1, cfg.batch_size])).astype(dtype=np.bool)
                else:
                    mask = np.equal(np.reshape(labels, [cfg.batch_size, 1]), np.reshape(labels, [1, cfg.batch_size]))

                summary, _, _ = session.run(
                    [merged, model.assignment, step],
                    {
                        model.t_images: feed_dict["images"],
                        model.prob: 0.5,
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

        self.RotationSSH(directory)
        self.RotationITQ(directory)
        self.RotationSITQ(directory)
        self.RotationRandomSearch(directory)

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

        longints = self.and_mode == 1

        lmdb_file = "./data/mirf" if self.cfg.dataset == "mirflickr" else None

        self.l_train, self.b_train = gen_hashes(model.t_images, model.prob, model.t_labels,
                                                model.output, session, items_train, hash_size, longints=longints, imagenet=self.cfg.dataset == "imagenet",lmdb_file = lmdb_file)

        self.l_test, self.b_test = gen_hashes(model.t_images, model.prob, model.t_labels,
                                              model.output, session, items_test, hash_size, 1, longints=longints, imagenet=self.cfg.dataset == "imagenet",lmdb_file = lmdb_file)

        if len(items_db) > 0:
            self.l_db, self.b_db = gen_hashes(model.t_images, model.prob, model.t_labels,
                                              model.output, session, items_db, hash_size, longints=longints, imagenet=self.cfg.dataset == "imagenet",lmdb_file = lmdb_file)
        else:
            self.l_db, self.b_db = self.l_train, self.b_train

        self.logger.info("Finished generating hashes")

        map_train, map_test = self.eval(directory, self.l_train, self.b_train, self.l_test, self.b_test, self.l_db, self.b_db)

        self.FAcc = map_test

    def RotationSSH(self, directory):
        self.logger.info("Starting rotations")
        labels = self.l_train
        H = self.b_train

        size = labels.shape[0]

        if size > 25000:
            idx = np.random.randint(size, size=25000)
            size = 25000
            labels = labels[idx,:]
            H = H[idx,:]

        if self.and_mode == 1 or self.and_mode == 2:
            S = np.bitwise_and(np.reshape(labels, [size, 1]),
                               np.reshape(labels, [1, size])).astype(dtype=np.bool)
        else:
            S = np.equal(np.reshape(labels, [size, 1]), np.reshape(labels, [1, size]))


        S = S * 2.0 - 1.0

        eta = 0.3

        M = np.matmul(np.matmul(H.T, S), H) + eta * np.matmul(H.T, H)

        U, s, Vh = np.linalg.svd(M, full_matrices=False)

        R = Vh

        b_train_r = np.matmul(self.b_train, R)
        b_test_r = np.matmul(self.b_test, R)
        b_db_r = np.matmul(self.b_db, R)
        self.logger.info("Finished ITQ rotations")

        self.eval(directory, self.l_train, b_train_r, self.l_test, b_test_r, self.l_db, b_db_r, "SSH")
        return

    def RotationITQ(self, directory):
        self.logger.info("Starting rotations")
        labels = self.l_train
        H = self.b_train

        size = labels.shape[0]

        if size > 25000:
            idx = np.random.randint(size, size=25000)
            H = H[idx, :]

        R = np.eye(self.cfg.hash_size, self.cfg.hash_size, dtype=np.float32)
        for i in range(500):
            #update B
            B = np.sign(np.matmul(H, R))

            #update R
            U, s, Vh = np.linalg.svd(np.matmul(B.T, H), full_matrices=False)
            R = np.matmul(Vh.T, U.T)

        b_train_r = np.matmul(self.b_train, R)
        b_test_r = np.matmul(self.b_test, R)
        b_db_r = np.matmul(self.b_db, R)
        self.logger.info("Finished ITQ rotations")

        self.eval(directory, self.l_train, b_train_r, self.l_test, b_test_r, self.l_db, b_db_r, "ITQ")
        return

    def RotationSITQ(self, directory):
        self.logger.info("Starting SITQ rotations")
        labels = self.l_train
        H = self.b_train

        size = labels.shape[0]

        if size > 25000:
            idx = np.random.randint(size, size=25000)
            size = 25000
            labels = labels[idx, :]
            H = H[idx,:]

        if self.and_mode == 1 or self.and_mode == 2:
            S = np.bitwise_and(np.reshape(labels, [size, 1]),
                               np.reshape(labels, [1, size])).astype(dtype=np.bool)
        else:
            S = np.equal(np.reshape(labels, [size, 1]), np.reshape(labels, [1, size]))

        S = S * 2.0 - 1.0

        R = np.eye(self.cfg.hash_size, self.cfg.hash_size, dtype=np.float32)
        for i in range(100):
            #update B
            B = np.matmul(S, np.sign(np.matmul(H, R)))

            #update R
            U, s, Vh = np.linalg.svd(np.matmul(B.T, H), full_matrices=False)
            R = np.matmul(Vh.T, U.T)

        b_train_r = np.matmul(self.b_train, R)
        b_test_r = np.matmul(self.b_test, R)
        b_db_r = np.matmul(self.b_db, R)
        self.logger.info("Finished rotations")

        self.eval(directory, self.l_train, b_train_r, self.l_test, b_test_r, self.l_db, b_db_r, "SITQ")
        return

    def RotationRandomSearch(self, directory):
        self.logger.info("Starting RandomSearch rotations")
        labels = np.array(self.l_train)
        H = self.b_train.astype(np.float32)

        size = labels.shape[0]

        idx = np.random.permutation(size)

        labels_q = labels
        labels_db = labels
        H_q = H
        H_db = H

        if size > 18000:
            idx_q = np.copy(idx[:2000])
            idx_db = np.copy(idx[2000:][:16000])
            labels_q = labels[idx_q,:]
            labels_db = labels[idx_db,:]
            H_q = H[idx_q, :]
            H_db = H[idx_db, :]
        elif size > 5000:
            idx_q = np.copy(idx[:2000])
            idx_db = np.copy(idx[2000:])
            labels_q = labels[idx_q,:]
            labels_db = labels[idx_db,:]
            H_q = H[idx_q, :]
            H_db = H[idx_db, :]

        print("DB size: %d Query set size: %d" % (H_db.shape[0], H_q.shape[0]))

        R = np.eye(self.cfg.hash_size, self.cfg.hash_size, dtype=np.float32)

        mapd0 = compute_map_fast(H_db, H_q, labels_db, labels_q, and_mode=self.and_mode==1, weighted_mode = self.and_mode == 2)
        step = 1.0

        worker_count = 1
        steps = int(800 / worker_count)
        results = [(0, np.eye(self.cfg.hash_size, self.cfg.hash_size, dtype=np.float32)) for i in range(worker_count)]

        for i in range(steps):
            step = (steps - i) / steps

            def ComputeNewValue(w):
                rBasis = random_rotation(self.cfg.hash_size).astype(np.float32)
                if random() > 0.5:
                    s = step
                else:
                    s = -step

                deltaR = np.matmul(rBasis.T, np.matmul(GetBaseRotation(s, self.cfg.hash_size), rBasis))
                newR = np.matmul(R, deltaR)
                rotated_data = np.matmul(H_db, newR)
                rotated_data_q = np.matmul(H_q, newR)
                mapd1 = compute_map_fast(rotated_data, rotated_data_q, labels_db, labels_q, and_mode=self.and_mode==1,weighted_mode = self.and_mode == 2)
                results[w] = (mapd1, newR)

            threads = []
            for w in range(worker_count):
                t = threading.Thread(target=ComputeNewValue, args=(w,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            updated = False
            for w in range(worker_count):
                #print("%f " % results[w][0], end='')
                if results[w][0] > mapd0:
                    R = results[w][1]
                    mapd0 = results[w][0]
                    updated = True
            print("")
            if updated:
                print("++++++++++++++ %f ++++++++++++++++" % mapd0)

        b_train_r = np.matmul(self.b_train, R)
        b_test_r = np.matmul(self.b_test, R)
        b_db_r = np.matmul(self.b_db, R)
        self.logger.info("Finished rotations")

        self.eval(directory, self.l_train, b_train_r, self.l_test, b_test_r, self.l_db, b_db_r, "RandomSearch")
        return

    def eval(self, directory, l_train, b_train, l_test, b_test, l_db, b_db, prefix="No rotation"):
        self.logger.info("Starting evaluation")
        map_train, map_test, curve = evaluate(
              l_train
            , b_train
            , l_test
            , b_test
            , l_db
            , b_db
            , top_n=self.top_n
            , and_mode=self.and_mode == 1
            , force_slow=self.and_mode == 1
            , weighted_mode = self.and_mode == 2)

        report_string = prefix + ": Test on train: {0}; Test on test: {1}".format(map_train, map_test)

        with open(os.path.join(directory, "results.txt"), "a") as file:
            file.write(report_string + "\n")
        self.logger.info(report_string)

        if curve is not None:
            output = open(os.path.join(directory, "pr_curve.pkl"), 'wb')
            pickle.dump(curve, output)
            output.close()

        return map_train, map_test
