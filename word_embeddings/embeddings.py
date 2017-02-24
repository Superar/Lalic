# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import re
import collections
import random
import math
import tempfile
import matplotlib.pyplot as plt

vocabulary_size = 50000
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2


# Lê os dados com tokens simples
def read_data(path):
    with open(path, 'r') as arquivo:
        return re.split('\W+', arquivo.read().lower(), flags=re.UNICODE)


# Criação do dataset
def cria_dataset(words):
    # Cria contagem das palavras mais frequentes em words
    count = [['UKN', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    # Cria vocabulário com a palavra e o índice dela no vetor count
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    # Cria lista que representa todas as palavras com os respectivos índices
    # Também conta quantas palavras não estão no vocabulário (dict)
    data = list()
    ukn_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            ukn_count += 1
        data.append(index)

    count[0][1] = ukn_count  # Atualiza contagem do token 'UKN'
    # Cria um dicionário para retornar a palavra com o índice
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reverse_dictionary


words = read_data('../Corpus_FAPESP_pt-en_bitexts/fapesp-bitexts.pt-en.pt')
data, count, vocab, reverse_vocab = cria_dataset(words)
del words
print('Dados carregados')

# Leitura do texto com janelas de contexto
data_index = 0


def cria_batch(tam_batch, num_skips, skip_window):
    global data_index
    assert tam_batch % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=tam_batch, dtype=np.int32)
    labels = np.ndarray(shape=(tam_batch, 1), dtype=np.int32)

    span = 2 * skip_window + 1  # [ skip_window palavra skip_window ]
    buff = collections.deque(maxlen=span)

    # Preenche o buffer com os índices das palavras no vocabulário
    for _ in range(span):
        buff.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # Gera dados relacionados a cada centro do buffer
    # Escolhe como target num_skip palavras aleatórias dentro do contexto
    for i in range(tam_batch // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buff[skip_window]
            labels[i * num_skips + j, 0] = buff[target]
        buff.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


# Cria um conjunto de validação aleatório com 16 das 100 palavras mais frequentes
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

# Criação do grafo

graph = tf.Graph()

with graph.as_default():
    # Dados de entrada
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size], name="train_inputs")
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1], name="train_labels")
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32, name="valid_dataset")

    # Variável a ser estimada: Todos os embeddings de todas as palavras no vocabulário
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="embeddings")
    # Utilizado para observar apenas as embeddings dos dados em train_inputs, serve para poupar custo
    embed = tf.nn.embedding_lookup(embeddings, train_inputs, name="embed")

    # Normalização dos embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    # Peso e bias para a rede
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                  stddev=1.0 / math.sqrt(embedding_size)), name="nce_weights")
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="nce_biases")

    # Loss function NCE
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    init = tf.global_variables_initializer()

    num_steps = 10

    with tf.Session(graph=graph) as session:
        init.run()
        print('Grafo criado')

        # Criação do log para salvar os passos do treinamento e vizualizar no TensorBoard
        logdir = tempfile.mkdtemp()
        print(logdir)
        summary_op = tf.summary.scalar("loss", loss)
        summary_writer = tf.summary.FileWriter(logdir, session.graph)

        for step in xrange(num_steps):
            batch_inputs, batch_labels = cria_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            loss_val, summary, _ = session.run([loss, summary_op, optimizer], feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)

        final_embeddings = normalized_embeddings.eval()

        summary_writer.flush()

        print('Treinamento concluído')


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate('%s' % label.decode('cp860'),
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)

try:
    from sklearn.manifold import TSNE

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_vocab[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print('Please install sklearn, matplotlib, and scipy to visualize embeddings.')
