# Copyright 2017 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Batch provider. Returns iterator to batches"""

from random import shuffle
# import matplotlib.pyplot as plt
from scipy import misc
try:
    import queue
except ImportError:
    import Queue as queue
from threading import Thread, Lock, Event
import logging

class BatchProvider:
    """All in memory batch provider for small datasets that fit RAM"""
    def __init__(self, batch_size, items, cycled=True, width=224, height=224):
        self.items = items
        shuffle(self.items)
        self.batch_size = batch_size

        self.current_batch = 0
        self.cycled = cycled
        self.done = False
        self.image_size = (width, height)
        self.lock = Lock()
        self.quit_event = Event()

        self.q = queue.Queue(16)
        self.batches_n = len(self.items)//self.batch_size
        logging.debug("Batches per epoch: {0}", self.batches_n)

    def get_batches(self):
        workers = []
        for i in range(16):
            worker = Thread(target=self._worker)
            worker.setDaemon(True)
            worker.start()
            workers.append(worker)
        try:
            while True:
                yield self._get_batch()

        except GeneratorExit:
            self.quit_event.set()
            self.done = True
            while not self.q.empty():
                try:
                    self.q.get(False)
                except queue.Empty:
                    continue
                self.q.task_done()

    def _worker(self):
        while not (self.quit_event.is_set() and self.done):
            b = self.__next()
            if b is None:
                break
            self.q.put(b)

    def _get_batch(self):
        if self.q.empty() and self.done:
            return None
        item = self.q.get()
        self.q.task_done()
        return item

    def __next(self):
        self.lock.acquire()
        if self.current_batch == self.batches_n:
            self.done = True
            if self.cycled:
                self.done = False
                self.current_batch = 0
                shuffled = list(self.items)
                shuffle(shuffled)
                self.items = shuffled
            else:
                self.lock.release()
                return None
        cb = self.current_batch
        self.current_batch += 1
        items = self.items
        self.lock.release()

        b_images = []
        b_labels = []
        
        for i in range(self.batch_size):
            item = items[cb * self.batch_size + i]

            image = misc.imresize(item[1], self.image_size, interp='bilinear')
            # Data augmentation. Should be removed from here
            # if random.random() > 0.5:
            #    im = np.fliplr(image)
            # image = np.roll(im, random.randint(-21, 21), 0)
            # image = np.roll(im, random.randint(-21, 21), 1)
            # plt.imshow(image)
            # plt.show()

            b_images.append(image)
            b_labels.append([item[0]])
        feed_dict = {"images": b_images, "labels": b_labels}

        return feed_dict


# For testing
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.cifar10_reader import Reader
    
    r = Reader('data/cifar-10-batches-bin')

    p = BatchProvider(5, r.items)

    b = p.get_batches()

    ims = next(b)["images"]
    for im in ims:
        plt.imshow(im, interpolation='nearest')
        plt.show()
