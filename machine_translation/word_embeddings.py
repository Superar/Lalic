# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import re
import collections
import random
import math


# TODO: Testar classe Embedder


class Embedder(object):
    """Gerador de word embeddings segundo o modelo skip-gram."""

    def __init__(self, options, num_skips, skip_window,
                 data, dictionary, reverse_dictionary):
        # Configurações
        self._options = options
        self._num_skips = num_skips
        self._skip_window = skip_window
        self._data = data
        self._dict = dictionary
        self._rev_dict = reverse_dictionary

        # Variáveis "globais"
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

    def _cria_grafo(self):
        opts = self._options

        self._train_inputs = tf.placeholder(tf.int32, shape=[batch_size],
                                            name="train_inputs")
        self._train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1],
                                            name="train_labels")
        self._embeddings = tf.Variable(tf.random_uniform([opts.vocab_size, opts.embedding_dim], -1.0, 1.0),
                                       name="embeddings")
        self._embed = tf.nn.embedding_lookup(self._embeddings, self._train_inputs,
                                             name="embed")
        self._norm = tf.sqrt(tf.reduce_sum(tf.square(self._embeddings), 1, keep_dims=True))
        self._normalized_embeddings = self._embeddings / self._norm
        self._nce_weights = tf.Variable(tf.truncated_normal([opts.vocab_size, opts.embedding_dim],
                                                            stddev=1.0 / math.sqrt(opts.embedding_dim)),
                                        name="nce_weights")
        self._nce_biases = tf.Variable(tf.zeros([opts.vocab_size]),
                                       name="nce_biases")

        # Otimização
        self._loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=self._nce_weights,
                           biases=self._nce_biases,
                           labels=self._train_labels,
                           inputs=self._embed,
                        #    num_sampled=num_sampled,
                           num_classes=opts.vocab_size))

        self._optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self._loss)

    def create_embeddings(self, num_steps):
        with tf.Graph().as_default(), tf.Session() as session:
            self._cria_grafo()
            session.run(tf.global_variables_initializer())

            for _ in xrange(num_steps):
                batch_inputs, batch_labels = self._cria_batch()
                feed_dict = {self._train_inputs: batch_inputs, self._train_labels: batch_labels}

                loss_val, _ = session.run([loss, optimizer], feed_dict=feed_dict)

        final_embeddings = self._normalized_embeddings.eval()

        return final_embeddings
