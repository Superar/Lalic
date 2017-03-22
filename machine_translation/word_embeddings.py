# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import re
import collections
import random
import math

class Embedder(object):
    """Gerador de word embeddings segundo o modelo skip-gram."""

    def __init__(self, options, num_skips, skip_window,
                 data, dictionary, reverse_dictionary):
        self._options = options
        self._num_skips = num_skips
        self._skip_window = skip_window
        self._data = data
        self._dict = dictionary
        self._rev_dict = reverse_dictionary

        self._data_index = 0

    def _cria_batch(self):
        assert tam_batch % num_skips == 0
        assert num_skips <= 2 * skip_window

        opts = self._options

        batch = np.ndarray(shape=opts.batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(opts.batch_size, 1), dtype=np.int32)

        span = 2 * self._skip_window + 1
        buff = collections.deque(maxlen=span)

        for _ in range(span):
            buff.append(self._data[self._data_index])
            self._data_index = (self._data_index + 1) % len(self._data)

        for i in range(opts.batch_size // self._num_skips):
            target = self._skip_window
            targets_to_avoid = [self._skip_window]
            for j in range(self._num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self._num_skips + j] = buff[self._skip_window]
                labels[i * self._num_skips + j, 0] = buff[target]
            buff.append(self._data[self._data_index])
            self._data_index = (self._data_index + 1) % len(self._data)
        return batch, labels
