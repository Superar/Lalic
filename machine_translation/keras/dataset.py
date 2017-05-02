import numpy as np
import codecs
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing import text


class Dataset(object):
    """Conjunto de Dados de um tradutor"""

    def __init__(self, options):
        self.options = options

        text_pt = self._read_data(options.path_pt)
        self.data_pt, self.dict_pt, self.rev_dict_pt = self._create_dataset(text_pt)
        text_en = self._read_data(options.path_en)
        self.data_en, self.dict_en, self.rev_dict_en = self._create_dataset(text_en)

        self.data_pt = sequence.pad_sequences(self.data_pt,
                                              maxlen=options.sequence_length)
        self.data_en = sequence.pad_sequences(self.data_en,
                                              maxlen=options.sequence_length)
        self.data_en = np.reshape(self.data_en,
                                  (-1, options.sequence_length, 1))


    @staticmethod
    def _read_data(path):
        with codecs.open(path, encoding='utf-8') as _file:
            return _file.read()


    def _create_dataset(self, input_text):
        tokenizer = Tokenizer(num_words=self.options.vocabulary_size)
        tokenizer.fit_on_texts([input_text])
        sequences = tokenizer.texts_to_sequences([input_text])

        index = 0
        _sequences = list()
        for __ in range(len(sequences[0]) // self.options.sequence_length):
            _sequences.append(sequences[0][index : index + self.options.sequence_length])
            index = index + self.options.sequence_length
        _sequences.append(sequences[0][index:])
        sequences = sequence.pad_sequences(_sequences, maxlen=self.options.sequence_length)

        word_index = tokenizer.word_index
        rev_word_index = dict(zip(word_index.values(), word_index.keys()))

        return sequences, word_index, rev_word_index
