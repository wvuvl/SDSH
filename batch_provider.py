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
import queue
from threading import Thread, Lock

class BatchProvider:
    """All in memory batch provider for small datasets that fit RAM"""
    def __init__(self, batch_size, items, cycled=True, width=224, height=224):
        self.items = items
        self.batch_size = batch_size

        self.current_image = 0
        self.cycled = cycled
        self.done = False
        self.image_size = (width, height)
        self.mutex = Lock()

        self.batch = None

    def get_batches(self):
        t = Thread(target=self._get_batch)
        t.start()
        while True:
            t.join()
            b = self.batch
            t = Thread(target=self._get_batch)
            t.start()
            yield b

    def _get_batch(self):
        """Return batch generator. Batch is  a dict {"images": <images>, "labels": <labels>}"""

        b_images = []
        b_labels = []
        for i in range(0, self.batch_size):
            if self.current_image == len(self.items):
                if self.cycled:
                    self.current_image = 0
                    shuffle(self.items)
                else:
                    self.batch = None
                    return None
            ci = self.current_image
            self.current_image += 1
            image = misc.imresize(self.items[ci][1], self.image_size, interp='bilinear')
            # Data augmentation. Should be removed from here
            # if random.random() > 0.5:
            #    im = np.fliplr(image)
            # image = np.roll(im, random.randint(-21, 21), 0)
            # image = np.roll(im, random.randint(-21, 21), 1)
            # plt.imshow(image)
            # plt.show()
            b_images.append(image)
            b_labels.append([self.items[ci][0]])
        feed_dict = {"images": b_images, "labels": b_labels}

        self.batch = feed_dict
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
