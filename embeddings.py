import tensorflow as tf
import numpy as np
import re
import collections

vocabulary_size = 50000
embedding_size = 10
batch_size = 100
num_sampled = 1000

# Lê os dados com tokens simples
def read_data(path):
	with open(path, 'r') as arquivo:
		return re.split('\W+', arquivo.read().lower())


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

	count[0][1] = ukn_count # Atualiza contagem do token 'UKN'
	# Cria um dicionário para retornar a palavra com o índice
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

	return data, count, dictionary, reverse_dictionary

words = read_data('Corpus_FAPESP_pt-en_bitexts/fapesp-bitexts.pt-en.pt')
data, count, dictionary, reverse_dictionary = cria_dataset(words)
del(words)


# Leitura do texto com janelas de contexto
data_index = 0

def cria_batch(batch_size, num_skips, skip_windows):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_windows
	
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

	span = 2 * skip_window + 1
	buffer = collections.deque(maxlen=span)

	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	
	for i in range(batch_size // num_skips):
		target = skip_window
		targets_to_avoid = [skip_window]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels

'''embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

nce_weights = tf.Variable(
	tf.truncated_normal([vocabulary_size, embedding_size],
						stddev=1.0 / math.sqrt(embedding_size)))

nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

loss = tf.reduce_mean(
	tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
				num_sampled, vocabulary_size))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss'''

with open('Corpus_FAPESP_pt-en_bitexts/fapesp-bitexts.pt-en.pt') as arquivo:
	cria_data_set()
