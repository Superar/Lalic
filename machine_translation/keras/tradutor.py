import numpy as np
import codecs
import argparse
import re
from collections import Counter
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

parser = argparse.ArgumentParser()
parser.add_argument('-pt', '--path_pt', help='Caminho para texto em português',
                    type=str, default=None)
parser.add_argument('-en', '--path_en', help='Caminho para texto em inglês',
                    type=str, default=None)
parser.add_argument('-vsize', '--vocabulary_size', help='Tamanho do vocabulário',
                    type=int, default=50000)
parser.add_argument('-esize', '--embedding_size', help='Dimensões para as word embeddings',
                    type=int, default=64)
parser.add_argument('-hsize', '--hidden_size', help='Dimensões para as redes neurais',
                    type=int, default=512)
parser.add_argument('-slen', '--sequence_length', help='Tamanho máximo para as sequências',
                    type=int, default=128)

FLAGS = parser.parse_args()

def read_data(path):
    with codecs.open(FLAGS.path_pt, encoding='utf-8') as _file:
        return re.split('\W+', _file.read().lower(), flags=re.UNICODE)


def create_dataset(words):
    count = [['<IGN>', -1], ['<START>', -1], ['END', -1], ['UKN', -1]]
    count.extend(Counter(words).most_common(FLAGS.vocabulary_size - 1))

    dictionary = dict()
    for word, __ in count:
        dictionary[word] = len(dictionary)

    data = list()
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
        data.append(index)

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return np.array(data), dictionary, reverse_dictionary

if not FLAGS.path_pt or not FLAGS.path_en:
    raise ValueError('--path_pt e --path_en são necessários.')
else:
    data_pt, dict_pt, rev_dict_pt = create_dataset(read_data(FLAGS.path_pt))
    data_en, dict_en, rev_dict_en = create_dataset(read_data(FLAGS.path_en))

    hidden_size = 512

    model = Sequential()
    model.add(Embedding(FLAGS.vocabulary_size,
                        FLAGS.embedding_size,
                        input_length=FLAGS.sequence_length,
                        mask_zero=True))
    model.add(GRU(hidden_size,
                  input_shape=(FLAGS.sequence_length, FLAGS.embedding_size)))
    model.add(Dense(FLAGS.hidden_size,
                    input_shape=(FLAGS.hidden_size,)))
    model.add(Activation('relu'))
    model.add(RepeatVector(FLAGS.sequence_length,
                           input_shape=(None, FLAGS.hidden_size)))
    model.add(GRU(FLAGS.hidden_size,
                  input_shape=(None, FLAGS.hidden_size),
                  return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='softmax'),
                              input_shape=(FLAGS.hidden_size, FLAGS.vocabulary_size)))
    model.compile(optimizer='adam', loss='mse')

    model.fit(data_pt, data_en,
              batch_size=64, epochs=3)

    model.reset_states()

    scores = model.evaluate(data_pt, data_en)
