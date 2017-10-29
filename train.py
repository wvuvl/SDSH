#! python3
import logging
import os
import pickle
import sys
import time
from pprint import pformat

import tensorflow as tf
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

        cfg = Cfg()

        for key in config:
            setattr(cfg, key, config[key])

        name = "{0}_{1}_{2}_{3}".format(cfg.loss, cfg.hash_size, cfg.margin, cfg.weight_decay_factor)

        directory = os.path.join(path, name)
        self.directory = directory

        logging.info("Starting {0}...".format(name))

        if not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.exists(os.path.join(directory, "Done.txt")):
            logging.info("\tWas already finished, skipping {0}".format(name))
            return

        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        self.logger = logger

        file_handler = logging.FileHandler(os.path.join(directory, name + ".log"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(self.formatter)

        logger.addHandler(file_handler)
        logger.addHandler(self.console_handler)

        with tf.Graph().as_default(), tf.Session() as session:
            logger.info("\n{0}\n{1}\n{0}\n".format("-" * 80, name))
            logger.info("\nSettings:\n{0}".format(pformat(vars(cfg))))

            with open('temp/items_train.pkl', 'rb') as pkl:
                items_train = pickle.load(pkl)
            with open('temp/items_test.pkl', 'rb') as pkl:
                items_test = pickle.load(pkl)

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

            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(cfg.learning_rate,
                                            global_step,
                                            decay_steps,
                                            cfg.learning_rate_decay_factor,
                                            staircase=True)
            tf.summary.scalar('learning_rate', lr)

            opt = tf.train.GradientDescentOptimizer(lr)

            train_step = opt.minimize(model.loss, global_step=global_step)
            _start_time = time.time()
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(directory, flush_secs=10, graph=session.graph)

            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)

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
                summary, _, _ = session.run(
                    [merged, model.assignment, train_step],
                    {
                        model.t_images: feed_dict["images"],
                        model.t_labels: feed_dict["labels"]
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
                    self.TestAndSaveCheckpoint(model, session, items_train, items_test, cfg.hash_size,
                                               directory, embedding_conf, saver, global_step, feed_dict)

            self.TestAndSaveCheckpoint(model, session, items_train, items_test, cfg.hash_size,
                                       directory, embedding_conf, saver, global_step)

            with open(os.path.join(directory, "Done.txt"), "a") as file:
                file.write("\n")

    def TestAndSaveCheckpoint(self, model, session, items_train, items_test, hash_size,
                              directory, embedding_conf, saver, global_step, feed_dict=None):
        saver.save(session, os.path.join(directory, "checkpoint"), global_step)

        if feed_dict is not None:
            file = open(os.path.join(directory, embedding_conf.metadata_path), "w")
            for l in feed_dict["labels"]:
                file.write(str(l[0]) + "\n")
            file.close()

        self.logger.info("Start generating hashes")
        l_dataset, b_dataset, l_test, b_test = gen_hashes(model.t_images, model.t_labels, model.output,
                                       session, items_train, items_test, hash_size)

        self.logger.info("Finished generating hashes")
        self.logger.info("Starting evaluation")

        map_train, map_test = evaluate(l_dataset, b_dataset, l_test, b_test)

        with open(os.path.join(directory, "results.txt"), "a") as file:
            file.write(str(map_train) + "\t" + str(map_test) + "\n")
        self.logger.info("Test on train: {0}, Test on test: {1}".format(map_train, map_test))
