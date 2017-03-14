import re
import collections
import numpy as np
import tensorflow as tf
import tempfile
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn.python.ops import rnn_cell
import os


# IDEA: Transformar em classe

def read_data(path):
    with open(path, 'r') as arquivo:
        return re.split('\W+', arquivo.read().lower(), flags=re.UNICODE)


# Criação do dataset
def cria_dataset(words, vocabulary_size):
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


vocab_size = 50000
seq_length = 128
batch_size = 128
embedding_dim = 128
memory_dim = 100
learning_rate = 0.05
momentum = 0.9

texto_portugues = read_data('../Corpus_FAPESP_pt-en_bitexts/fapesp-bitexts.pt-en.pt')
data_pt, dict_pt, rev_dict_pt = cria_dataset(texto_portugues, vocab_size)
print('Vocabulário português extraído')

texto_ingles = read_data('../Corpus_FAPESP_pt-en_bitexts/fapesp-bitexts.pt-en.en')
data_en, dict_en, rev_dict_en = cria_dataset(texto_ingles, vocab_size)
print('Vocabulário inglês extraído')

enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                          name="inp%i" % t)
           for t in range(seq_length)]
labels = [tf.placeholder(tf.int32, shape=(None,),
                         name="labels%i" % t)
          for t in range(seq_length)]
weights = [tf.ones_like(labels_t, dtype=tf.float32)
           for labels_t in labels]
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")]
          + enc_inp[:-1])
prev_mem = tf.zeros((batch_size, memory_dim))
print('Tensors criados')

sess = tf.InteractiveSession()
cell = tf.contrib.rnn.LSTMCell(memory_dim)
dec_outputs, dec_memory = seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_dim)
loss = seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)
print('Modelo criado')

optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)
print('Otimizador gerado')

summary_op = tf.summary.scalar("loss", loss)
magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1])))
summary_magnitude = tf.summary.scalar("magnitude at t=1", magnitude)
logdir = tempfile.mkdtemp()
summary_writer = tf.summary.FileWriter(logdir, sess.graph)
print('Tensorboard iniciado em ' + logdir)

sess.run(tf.global_variables_initializer())
print('Iniciando treinamento')

data_index = 0
for t in range(450):

    x = []
    y = []

    for _ in range(batch_size // seq_length):
        x.append(data_pt[data_index:data_index+seq_length])
        y.append(data_en[data_index:data_index+seq_length])
        data_index = data_index + seq_length

    x = np.array(x).T
    y = np.array(y).T

    feed_dict = {enc_inp[t]: x[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: y[t] for t in range(seq_length)})

    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    summary_writer.add_summary(summary, t)

summary_writer.flush()

if not os.path.exists('tradutor'):
    os.mkdir('tradutor')

saver = tf.train.Saver(tf.global_variables())
saver.save(sess, 'tradutor/modelo-encoder-decoder')

with open('tradutor/vocab_pt', 'w') as file_pt:
    file_pt.write('DATA\n')
    for word in data_pt:
        file_pt.write('{} '.format(word))
    file_pt.write('\nDICTIONARY\n')
    for i in dict_pt.keys():
        file_pt.write("{}@@{}\n".format(i, dict_pt[i]))
    file_pt.write('REVERSE_DICTIONARY\n')
    for i in rev_dict_pt.keys():
        file_pt.write("{}@@{}\n".format(i, rev_dict_pt[i]))

with open('tradutor/vocab_en', 'w') as file_en:
    file_en.write('DATA\n')
    for word in data_en:
        file_en.write('{} '.format(word))
    file_en.write('\nDICTIONARY\n')
    for i in dict_en.keys():
        file_en.write("{}@@{}\n".format(i, dict_en[i]))
    file_en.write('REVERSE_DICTIONARY\n')
    for i in rev_dict_en.keys():
        file_en.write("{}@@{}\n".format(i, rev_dict_en[i]))

print('Modelo salvo')
