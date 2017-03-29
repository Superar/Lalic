import re
import collections
import numpy as np
import tensorflow as tf
import tempfile
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn.python.ops import rnn_cell
import os
import codecs
import sys
from options import Options
from word_embeddings import Embedder

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Diretório para salvar o modelo")
flags.DEFINE_string("path_pt", None, "Caminho para texto em português")
flags.DEFINE_string("path_en", None, "Caminho para texto em inglês")
flags.DEFINE_integer("vocab_size", 50000, "Tamanho do vocabulário")
flags.DEFINE_integer("seq_length", 128, "Tamanho de cada sequência para computação")
flags.DEFINE_integer("batch_size", 128, "Tamanho de batch para computar")
flags.DEFINE_integer("embedding_dim", 128, "Dimensão dos vetores para representar as palavras")
flags.DEFINE_integer("memory_dim", 100, "Tamanho da memória das RNN")
flags.DEFINE_integer("iterations", 450, "Número de iterações para o treinamento")
flags.DEFINE_float("learning_rate", 0.05, "Learning rate para treinamento")
flags.DEFINE_float("momentum", 0.9, "Momentum para treinamento")
flags.DEFINE_boolean("load_model", False, "Indica se o modelo precisa ser carregado de um arquivo")

FLAGS = flags.FLAGS


class Dataset(object):
    """Conjunto de dados para o tradutor."""

    def __init__(self, options):
        self._options = options
        if options.load_model:
            self.data_pt, self.dict_pt, self.rev_dict_pt = self._recria_dataset(os.path.join(options.save_path, 'vocab_pt'))
            self.data_en, self.dict_en, self.rev_dict_en = self._recria_dataset(os.path.join(options.save_path, 'vocab_en'))
        else:
            texto_portugues = self._read_data(options.path_pt)
            self.data_pt, self.dict_pt, self.rev_dict_pt = self._cria_dataset(texto_portugues)
            texto_ingles = self._read_data(options.path_en)
            self.data_en, self.dict_en, self.rev_dict_en = self._cria_dataset(texto_ingles)

        self.save_vocab()

    @staticmethod
    def _read_data(path):
        with codecs.open(path, encoding='utf-8') as arquivo:
            return re.split('\W+', arquivo.read().lower(), flags=re.UNICODE)

    # Criação do dataset
    def _cria_dataset(self, words):
        """Lê e cria o conjunto de dados e vocabulário."""
        opts = self._options

        count = [['UKN', -1]]
        count.extend(collections.Counter(words).most_common(opts.vocab_size - 1))

        dictionary = dict()
        for word, __ in count:
            dictionary[word] = len(dictionary)

        data = list()
        ukn_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                ukn_count += 1
            data.append(index)

        count[0][1] = ukn_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        return data, dictionary, reverse_dictionary

    def save_vocab(self):
        """Salva o vocabulário para futuro carregamento."""
        opts = self._options

        if not os.path.exists(opts.save_path):
            os.mkdir(opts.save_path)

        with codecs.open(os.path.join(opts.save_path, 'vocab_pt'), 'w', encoding='utf-8') as file_pt:
            for word in self.data_pt:
                file_pt.write('{} '.format(word))
            file_pt.write('\n{}\n'.format(opts.vocab_size))
            for i in self.dict_pt.keys():
                file_pt.write("{}%@{}\n".format(i, self.dict_pt[i]))
            file_pt.write('{}\n'.format(opts.vocab_size))
            for i in self.rev_dict_pt.keys():
                file_pt.write("{}%@{}\n".format(i, self.rev_dict_pt[i]))

        with codecs.open(os.path.join(opts.save_path, 'vocab_en'), 'w', encoding='utf-8') as file_en:
            for word in self.data_en:
                file_en.write('{} '.format(word))
            file_en.write('\n{}\n'.format(opts.vocab_size))
            for i in self.dict_en.keys():
                file_en.write("{}%@{}\n".format(i, self.dict_en[i]))
            file_en.write('{}\n'.format(opts.vocab_size))
            for i in self.rev_dict_en.keys():
                file_en.write("{}%@{}\n".format(i, self.rev_dict_en[i]))

    def _recria_dataset(self, path):
        """Carrega o vocabulário salvo."""
        opts = self._options

        dictionary = {}
        reverse_dictionary = {}
        data = []

        with codecs.open(path, encoding='utf-8') as file_:
            data = file_.readline().split()
            dict_size = int(file_.readline())
            for _ in range(dict_size):
                split_line = re.split('%@', file_.readline().rstrip(), flags=re.UNICODE)
                dictionary[split_line[0]] = split_line[1]
            rev_dict_size = int(file_.readline())
            for _ in range(rev_dict_size):
                split_line = re.split('%@', file_.readline().rstrip(), flags=re.UNICODE)
                reverse_dictionary[int(split_line[0])] = split_line[1]
        return data, dictionary, reverse_dictionary


class Tradutor(object):
    """Modelo de tradutor encoder-decoder."""

    def __init__(self, options, dataset, embeddings, session):
        self._options = options
        self._session = session
        self._dataset = dataset
        self._embeddings = embeddings
        self.build_graph()
        print('Grafo criado')

        if options.load_model:
            self.saver.restore(session,
                               os.path.join(options.save_path, 'encoder-decoder.ckpt'))
            print('Modelo carregado')

# TODO: Transformar em dynamic_rnn

    def _seq2seq_func(self):
        """Função para o modelo de sequência com embeddings."""
        opts = self._options

        return tf.nn.dynamic_rnn(self._cell, self._encoder_inputs, dtype=tf.float32)

    def _optimize(self):
        """Função para gerar o otimizador do modelo."""
        opts = self._options

        optimizer = tf.train.MomentumOptimizer(opts.learning_rate, opts.momentum)
        train_op = optimizer.minimize(self._loss)
        self._train_op = train_op

    def _create_tensors(self):
        """Função para gerar os Tensors do modelo."""
        opts = self._options

        encoder_inputs = tf.placeholder(tf.float32,
                                        [None, opts.seq_length, opts.embedding_dim],
                                        name = 'enc_inp')
        labels = tf.placeholder(tf.float32,
                                [None, opts.seq_length, opts.embedding_dim],
                                name = 'labels')
        weights = tf.Variable(tf.truncated_normal([1, opts.embedding_dim],
                                                  stddev=0.1))
        # decoder_inputs = ([tf.zeros_like(encoder_inputs[0], dtype=np.int32, name="GO")]
        #                   + encoder_inputs[:-1])

        return encoder_inputs, labels, weights # , decoder_inputs

    def build_graph(self):
        """Criação do grafo para o tradutor."""
        opts = self._options

        encoder_inputs, labels, weights = self._create_tensors()
        self._encoder_inputs = encoder_inputs
        self._labels = labels
        self._weights = weights
        # self._decoder_inputs = decoder_inputs

        self._cell = tf.contrib.rnn.LSTMCell(opts.memory_dim)
        self._decoder_outputs, self._decoder_memory = self._seq2seq_func()
        self._decoder_outputs = tf.reshape(self._decoder_outputs, [-1, opts.embedding_dim])

        self._output = tf.nn.softmax(tf.matmul(self._decoder_outputs, self._weights))
        self._output = tf.reshape(self._output, [-1, opts.seq_length, opts.embedding_dim])

        loss = tf.reduce_mean(-tf.reduce_sum(self._labels * tf.log(self._output), [1, 2]))
        self._loss = loss
        self._optimize()

        self._session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self):
        """Treinamento de acordo com o número de iterações."""
        opts = self._options

        data_index = 0
        for t in range(opts.iterations):

            x = []
            y = []

            for __ in range(opts.batch_size // opts.seq_length):
                x.append(self._dataset.data_pt[data_index:data_index + opts.seq_length])
                y.append(self._dataset.data_en[data_index:data_index + opts.seq_length])
                data_index = data_index + opts.seq_length

            x = np.array(x).T
            y = np.array(y).T

            feed_dict = {self._encoder_inputs[t]: x[t] for t in range(opts.seq_length)}
            feed_dict.update({self._labels[t]: y[t] for t in range(opts.seq_length)})

            _, loss_t = self._session.run([self._train_op, self._loss], feed_dict)

# TODO: Como tratar o caso do texto ser menor do que seq_length

    def _process_text(self, text):
        opts = self._options

        text_list = []
        tokens = re.split('\W+', text, flags=re.UNICODE)
        for t in tokens:
            if t not in self._dataset.dict_pt:
                text_list.append(self._dataset.dict_pt['UKN'])
            else:
                text_list.append((self._dataset.dict_pt[t]))

        text_length = len(text_list)
        if text_length < opts.seq_length:
            ukn_list = [self._dataset.dict_pt['UKN']] * (opts.seq_length - text_length)
            text_list.extend(ukn_list)
        return text_list

    def translate(self, text):
        opts = self._options

        processed_text = self._process_text(text)
        processed_text = np.array(processed_text).T

        feed_dict = {self._encoder_inputs[t]: processed_text[t].reshape((1,))
                     for t in range(opts.seq_length)}
        decoder_output = self._session.run(self._output, feed_dict)

        return decoder_output


def main(argv):
    if FLAGS.load_model:
        if not FLAGS.save_path:
            raise ValueError('--save_path é necessário')
        else:
            with tf.Graph().as_default(), tf.Session() as session_tradutor:
                opts = Options(FLAGS)
                dataset = Dataset(opts)
                embed = Embedder(opts, num_skips=2, skip_window=1,
                                 data=dataset.data_pt,
                                 dictionary=dataset.dict_pt,
                                 reverse_dictionary=dataset.rev_dict_pt)
                nmt = Tradutor(opts,
                               dataset=dataset,
                               embeddings=embed,
                               session=session_tradutor)
                print(nmt.translate('Oi oi testando não sei se vai dar certo isso'))
    else:
        if not FLAGS.path_pt or not FLAGS.path_en or not FLAGS.save_path:
            raise ValueError('--path_pt --path_en e --save_path são necessários.')
        else:
            grafo_tradutor = tf.Graph()
            with grafo_tradutor.as_default(), tf.Session() as session_tradutor:
                opts = Options(FLAGS)
                dataset = Dataset(opts)
                embed = Embedder(opts, num_skips=2, skip_window=1,
                                 data=dataset.data_pt,
                                 dictionary=dataset.dict_pt,
                                 reverse_dictionary=dataset.rev_dict_pt)
                nmt = Tradutor(opts,
                               dataset=dataset,
                               embeddings=embed,
                               session=session_tradutor)
                nmt.train()
                print('Modelo treinado com {} iterações'.format(FLAGS.iterations))
                nmt.saver.save(session_tradutor, os.path.join(opts.save_path, 'encoder-decoder.ckpt'))
                print('Modelo salvo')


if __name__ == '__main__':
    tf.app.run()
