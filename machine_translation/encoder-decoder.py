import re
import collections
import numpy as np
import tensorflow as tf
import tempfile
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn.python.ops import rnn_cell
import os
import codecs


class Tradutor(object):

    def __init__(self, load=False, path_dir_load=None,
                 path_pt=None, path_en=None, vocab_size=50000,
                 seq_length=128, batch_size=128, embedding_dim=128, memory_dim=100):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.memory_dim = memory_dim

        if load:
            assert path_dir_load
            self.data_pt, self.dict_pt, self.rev_dict_pt = self._recria_dataset(path_dir_load + '/vocab_pt')
            self.data_en, self.dict_en, self.rev_dict_en = self._recria_dataset(path_dir_load + '/vocab_en')
        else:
            assert path_pt and path_en
            texto_portugues = self._read_data(path_pt)
            self.data_pt, self.dict_pt, self.rev_dict_pt = self._cria_dataset(texto_portugues, vocab_size)

            texto_ingles = self._read_data(path_en)
            self.data_en, self.dict_en, self.rev_dict_en = self._cria_dataset(texto_ingles, vocab_size)


    def _read_data(self, path):
        with open(path, 'r') as arquivo:
            return re.split('\W+', arquivo.read().lower(), flags=re.UNICODE)


    # Criação do dataset
    def _cria_dataset(self, words, vocabulary_size):
        count = [['UKN', -1]]
        count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

        dictionary = dict()
        for word, _ in count:
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


    # Carrega dataset
    def _recria_dataset(self, path):
        with codecs.open(path, encoding='utf-8') as file_:
            dictionary = {}
            reverse_dictionary = {}
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


    def train(self, iterations=450, learning_rate=0.05, momentum=0.9, save=True):
        enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                                  name="inp%i" % t)
                   for t in range(self.seq_length)]
        labels = [tf.placeholder(tf.int32, shape=(None,),
                                 name="labels%i" % t)
                  for t in range(self.seq_length)]
        weights = [tf.ones_like(labels_t, dtype=tf.float32)
                   for labels_t in labels]
        dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")]
                  + enc_inp[:-1])
        prev_mem = tf.zeros((self.batch_size, self.memory_dim))
        print('Tensors criados')

        self.sess = tf.InteractiveSession()
        cell = tf.contrib.rnn.LSTMCell(self.memory_dim)
        self.dec_outputs, dec_memory = seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, self.vocab_size, self.vocab_size, self.embedding_dim)
        loss = seq2seq.sequence_loss(self.dec_outputs, labels, weights, self.vocab_size)
        print('Modelo criado')

        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        train_op = optimizer.minimize(loss)
        print('Otimizador gerado')

        if save:
            summary_op = tf.summary.scalar("loss", loss)
            magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1])))
            summary_magnitude = tf.summary.scalar("magnitude at t=1", magnitude)
            logdir = tempfile.mkdtemp()
            summary_writer = tf.summary.FileWriter(logdir, self.sess.graph)
            print('Tensorboard iniciado em ' + logdir)

        self.sess.run(tf.global_variables_initializer())
        print('Iniciando treinamento')

        data_index = 0
        for t in range(450):

            x = []
            y = []

            for __ in range(self.batch_size // self.seq_length):
                x.append(self.data_pt[data_index:data_index+self.seq_length])
                y.append(self.data_en[data_index:data_index+self.seq_length])
                data_index = data_index + self.seq_length

            x = np.array(x).T
            y = np.array(y).T

            feed_dict = {enc_inp[t]: x[t] for t in range(self.seq_length)}
            feed_dict.update({labels[t]: y[t] for t in range(self.seq_length)})

            if save:
                __, loss_t, summary = self.sess.run([train_op, loss, summary_op], feed_dict)
                summary_writer.add_summary(summary, t)
            else:
                __, loss_t = self.sess.run([train_op, loss], feed_dict)

        if save:
            summary_writer.flush()


    def salvar(self, path_dir='tradutor'):
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)

        saver = tf.train.Saver(tf.global_variables())
        tf.add_to_collection('dec_outputs', self.dec_outputs)
        saver.save(self.sess, path_dir + '/modelo-encoder-decoder')

        with open(path_dir + '/vocab_pt', 'w') as file_pt:
            file_pt.write('DATA\n')
            for word in self.data_pt:
                file_pt.write('{} '.format(word))
            file_pt.write('\nDICTIONARY\n')
            for i in self.dict_pt.keys():
                file_pt.write("{}@@{}\n".format(i, self.dict_pt[i]))
            file_pt.write('REVERSE_DICTIONARY\n')
            for i in self.rev_dict_pt.keys():
                file_pt.write("{}@@{}\n".format(i, self.rev_dict_pt[i]))

        with open(path_dir + '/vocab_en', 'w') as file_en:
            file_en.write('DATA\n')
            for word in self.data_en:
                file_en.write('{} '.format(word))
            file_en.write('\nDICTIONARY\n')
            for i in self.dict_en.keys():
                file_en.write("{}@@{}\n".format(i, self.dict_en[i]))
            file_en.write('REVERSE_DICTIONARY\n')
            for i in self.rev_dict_en.keys():
                file_en.write("{}@@{}\n".format(i, self.rev_dict_en[i]))


# nmt = Tradutor(path_pt='../Corpus_FAPESP_pt-en_bitexts/fapesp-bitexts.pt-en.pt', path_en='../Corpus_FAPESP_pt-en_bitexts/fapesp-bitexts.pt-en.en')
# nmt.train()
# print('Trainamento concluído')
# nmt.salvar()
# print('Modelo salvo')

nmt = Tradutor(load=True, path_dir_load='tradutor')
print(nmt.dict_en)
