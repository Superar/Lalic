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


class Options(object):
    """Opções do modelo do tradutor."""

    def __init__(self):

        # Opções para geração do vocabulário
        self.vocab_size = FLAGS.vocab_size
        self.path_pt = FLAGS.path_pt
        self.path_en = FLAGS.path_en

        # Opções para criação do modelo
        self.seq_length = FLAGS.seq_length
        self.batch_size = FLAGS.batch_size
        self.embedding_dim = FLAGS.embedding_dim
        self.memory_dim = FLAGS.memory_dim

        # Opções para treinamento
        self.iterations = FLAGS.iterations
        self.learning_rate = FLAGS.learning_rate
        self.momentum = FLAGS.momentum

        # Opções para salvar e carregar o modelo
        self.save_path = FLAGS.save_path
        self.load_model = FLAGS.load_model


class Tradutor(object):
    """Modelo de tradutor encoder-decoder."""

    def __init__(self, options, session):
        self._options = options
        self._session = session

# TODO: Arrumar o carregamento do modelo, dicionário está vazio

        if options.load_model:
            self.data_pt, self.dict_pt, self.rev_dict_pt = self._recria_dataset(os.path.join(options.save_path, 'vocab_pt'))
            self.data_en, self.dict_en, self.rev_dict_en = self._recria_dataset(os.path.join(options.save_path, 'vocab_en'))
            self.build_graph()
            print('Grafo criado')
            self.saver.restore(session, os.path.join(options.save_path, 'encoder-decoder.ckpt'))
            print('Modelo carregado')
        else:
            texto_portugues = self._read_data(options.path_pt)
            self.data_pt, self.dict_pt, self.rev_dict_pt = self._cria_dataset(texto_portugues)
            texto_ingles = self._read_data(options.path_en)
            self.data_en, self.dict_en, self.rev_dict_en = self._cria_dataset(texto_ingles)
            self.build_graph()
            print('Grafo criado')
            self.save_vocab()
            print('Vocabulário salvo')


    def _read_data(self, path):
        with open(path, 'r') as arquivo:
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


    def _seq2seq_func(self, enc_inp, dec_inp, feed):
        """Função para o modelo de sequência com embeddings."""
        opts = self._options

        return seq2seq.embedding_rnn_seq2seq(
                                  enc_inp, dec_inp, self._cell,
                                  opts.vocab_size, opts.vocab_size,
                                  opts.embedding_dim, feed_previous=feed)


    def _optimize(self, loss):
        """Função para gerar o otimizador do modelo."""
        opts = self._options

        optimizer = tf.train.MomentumOptimizer(opts.learning_rate, opts.momentum)
        train_op = optimizer.minimize(loss)
        self._train_op = train_op


    def _create_tensors(self):
        """Função para gerar os Tensors do modelo."""
        opts = self._options

        encoder_inputs = [tf.placeholder(tf.int32, shape=(None,),
                                               name='inp%i' % t)
                                for t in range(opts.seq_length)]
        labels = [tf.placeholder(tf.int32, shape=(None,),
                                  name='labels%i' % t)
                        for t in range(opts.seq_length)]
        weights = [tf.ones_like(labels_t, dtype=tf.float32)
                         for labels_t in labels]
        decoder_inputs = ([tf.zeros_like(encoder_inputs[0], dtype=np.int32, name="GO")]
                                + encoder_inputs[:-1])

        return encoder_inputs, labels, weights, decoder_inputs


    def build_graph(self):
        """Criação do grafo para o tradutor."""
        opts = self._options

        encoder_inputs, labels, weights, decoder_inputs = self._create_tensors()
        self._encoder_inputs = encoder_inputs
        self._labels = labels
        self._weights = weights
        self._decoder_inputs = decoder_inputs

        self._cell = tf.contrib.rnn.LSTMCell(opts.memory_dim)
        self._decoder_outputs, self._decorder_memory = self._seq2seq_func(encoder_inputs, decoder_inputs, False)

        loss = seq2seq.sequence_loss(self._decoder_outputs, labels, weights, opts.vocab_size)
        self._loss = loss
        self._optimize(loss)

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
                x.append(self.data_pt[data_index:data_index+opts.seq_length])
                y.append(self.data_en[data_index:data_index+opts.seq_length])
                data_index = data_index + opts.seq_length

            x = np.array(x).T
            y = np.array(y).T

            feed_dict = {self._encoder_inputs[t]: x[t] for t in range(opts.seq_length)}
            feed_dict.update({self._labels[t]: y[t] for t in range(opts.seq_length)})

            __, loss_t= self._session.run([self._train_op, self._loss], feed_dict)


    def save_vocab(self):
        opts = self._options

        if not os.path.exists(opts.save_path):
            os.mkdir(opts.save_path)

        with open(os.path.join(opts.save_path, 'vocab_pt'), 'w') as file_pt:
            file_pt.write('DATA\n')
            for word in self.data_pt:
                file_pt.write('{} '.format(word))
            file_pt.write('\nDICTIONARY\n')
            for i in self.dict_pt.keys():
                file_pt.write("{}@@{}\n".format(i, self.dict_pt[i]))
            file_pt.write('REVERSE_DICTIONARY\n')
            for i in self.rev_dict_pt.keys():
                file_pt.write("{}@@{}\n".format(i, self.rev_dict_pt[i]))

        with open(os.path.join(opts.save_path, 'vocab_en'), 'w') as file_en:
            file_en.write('DATA\n')
            for word in self.data_en:
                file_en.write('{} '.format(word))
            file_en.write('\nDICTIONARY\n')
            for i in self.dict_en.keys():
                file_en.write("{}@@{}\n".format(i, self.dict_en[i]))
            file_en.write('REVERSE_DICTIONARY\n')
            for i in self.rev_dict_en.keys():
                file_en.write("{}@@{}\n".format(i, self.rev_dict_en[i]))


    def _recria_dataset(self, path):
        with codecs.open(path, encoding='utf-8') as file_:
            dictionary = {}
            reverse_dictionary = {}
            data = []
            rev = False
            data_bool = False

            for line in file_.readlines():
                line = line.rstrip()
                if rev:
                    split_line = re.split('@@', line, flags=re.UNICODE)
                    reverse_dictionary[split_line[0]] = split_line[1]
                elif data_bool:
                    data = line.split()
                    data_bool = False
                elif line == 'DATA':
                    data_bool = True
                elif line == 'DICTIONARY':
                    rev = False
                elif line == 'REVERSE_DICTIONARY':
                    rev = True
                else:
                    split_line = re.split('@@', line, flags=re.UNICODE)
                    dictionary[split_line[0]] = split_line[1]

        return data, dictionary, reverse_dictionary


def main(argv):
    if FLAGS.load_model:
        if not FLAGS.save_path:
            raise ValueError('--save_path é necessário')
        else:
            with tf.Graph().as_default(), tf.Session() as session:
                opts = Options()
                nmt = Tradutor(opts, session)
                print(nmt.dict_pt)
    else:
        if not FLAGS.path_pt or not FLAGS.path_en or not FLAGS.save_path:
            raise ValueError('--path_pt --path_en e --save_path são necessários.')
        else:
            with tf.Graph().as_default(), tf.Session() as session:
                opts = Options()
                nmt = Tradutor(opts, session)
                nmt.train()
                print('Modelo treinado com {} iterações'.format(FLAGS.iterations))
                nmt.saver.save(session, os.path.join(opts.save_path, 'encoder-decoder.ckpt'))
                print('Modelo salvo')


if __name__ == '__main__':
    tf.app.run()
