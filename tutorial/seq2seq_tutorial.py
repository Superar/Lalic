# coding=utf-8

"""Baseado em
https://github.com/hans/ipython-notebooks/blob/master/tf/TF%20tutorial.ipynb
Para rodar:
python seq2seq_tutorial.py
tensorboard --logdir caminho/para/diretorio/temp"""

import numpy as np
import tensorflow as tf
import tempfile
# Não faz parte da API livre do TensorFlow
from tensorflow.python.ops import rnn_cell, seq2seq

# Constantes
seq_length = 5  # Tamanho da sequência
batch_size = 64

vocab_size = 7
embedding_dim = 50

memory_dim = 100

# Entrada do encoder
# Lista de tensors representando cada elemento da sequência
enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                          name="inp%i" % t)
           for t in range(seq_length)]

# Lista de labels representando cada elemento seguinte na sequência
labels = [tf.placeholder(tf.int32, shape=(None,),
                         name="labels%i" % t)
          for t in range(seq_length)]

# Lista de pesos (segundo cross-entropy) para cada instante de tempo
weights = [tf.ones_like(labels_t, dtype=tf.float32)
           for labels_t in labels]

# Entrada do decoder
# Inicia com um valor "GO" e continua com a frase
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")]
           + enc_inp[:-1])

# Valor inicial da memória da RNN
prev_mem = tf.zeros((batch_size, memory_dim))

# Criação do modelo
sess = tf.InteractiveSession()

cell = rnn_cell.GRUCell(memory_dim)

dec_outputs, dec_memory = seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_dim)

# Loss Function
loss = seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)

# Log para TesorBoard
summary_op = tf.summary.scalar("loss", loss)
magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1])))
summary_magnitude = tf.summary.scalar("magnitude at t=1", magnitude)
logdir = tempfile.mkdtemp()
print logdir
summary_writer = tf.summary.FileWriter(logdir, sess.graph)

# Otimização
learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)

sess.run(tf.global_variables_initializer())

# Treinamento com 500 iterações
for t in range(500):
    # Escolhe aleatóriamente seq_length valores de vocab_size sem repetição
    x = [np.random.choice(vocab_size, size=(seq_length,), replace=False)
         for _ in range(batch_size)]
    y = x[:]

    x = np.array(x).T
    y = np.array(y).T

    # Adiciona ao tensor os valores
    feed_dict = {enc_inp[t]: x[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: y[t] for t in range(seq_length)})

    # Inicia a sessão e retorna os valores para adicionar ao log
    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    summary_writer.add_summary(summary, t)

summary_writer.flush()
