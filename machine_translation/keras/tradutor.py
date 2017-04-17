import numpy as np
import codecs
import argparse
import re
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('-pt', '--path_pt', help='Caminho para texto em português',
                    type=str, default=None)
parser.add_argument('-en', '--path_en', help='Caminho para texto em inglês',
                    type=str, default=None)
parser.add_argument('-vsize', '--vocabulary_size', help='Tamanho do vocabulário',
                    type=int, default=50000)

FLAGS = parser.parse_args()

def read_data(path):
    with codecs.open(FLAGS.path_pt, encoding='utf-8') as _file:
        return re.split('\W+', _file.read().lower(), flags=re.UNICODE)


def create_dataset(words):
    count = [['UKN', -1]]
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
    print(data_pt, data_en)
