import numpy as np
import codecs
import re
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing import text


#TODO: Lidar com textos de tamanhos diferentes
# (Phrases = Colocar um tamanho máximo para as phrases e pegar todas as combinações
#  de possiblidades que cabem dentro daquele tamanho)
# (Truncar = Descartar palavras depois do tamanho máximo)
# (batch_size=1 no treinamento)
# (bucketing -> Cada linha tem uma sentença (alinhamento))

# Maior sentença português: 173
# Maior sentença inglês: 160

class Dataset(object):
    """Conjunto de Dados de um tradutor.

    Cada idioma possui os seguintes tipos de dados:

    data - Uma lista representando o texto, porém as palavras são
           representadas por índices.  As sentenças são representadas por
           listas de tamanho sequence_length.

    dictionary - Um dicionário relacionando as palavras aos
                 índices correspondentes

    rev_dictionary - Um dicionário relacionando os índices
                     às palavras correspondentes
    """

    def __init__(self, options):
        """ Contrutor da classe Dataset"""

        self.options = options

        if options.load:
            print('Carregando dataset...')
        else:
            text_pt = self._read_data(options.path_pt)
            self.data_pt, self.dict_pt, self.rev_dict_pt = self._create_dataset(text_pt)
            text_en = self._read_data(options.path_en)
            self.data_en, self.dict_en, self.rev_dict_en = self._create_dataset(text_en)

            self._save_vocab()

            self.data_pt = sequence.pad_sequences(self.data_pt,
                                                  maxlen=options.sequence_length)
            self.data_en = sequence.pad_sequences(self.data_en,
                                                  maxlen=options.sequence_length)
            self.data_en = np.reshape(self.data_en,
                                      (-1, options.sequence_length, 1))


    @staticmethod
    def _read_data(path):
        """Lê e retorna o texto de um arquivo em sentenças"""

        with codecs.open(path, encoding='utf-8') as _file:
            return _file.readlines()


    def _create_dataset(self, input_text):
        """Cria o conunto de dados a partir de um texto"""

        tokenizer = Tokenizer(num_words=self.options.vocabulary_size)
        tokenizer.fit_on_texts(input_text)
        sequences = tokenizer.texts_to_sequences(input_text)

        word_index = tokenizer.word_index
        rev_word_index = dict(zip(word_index.values(), word_index.keys()))

        return sequences, word_index, rev_word_index


    def _save_vocab(self):
        """Salva vocabulário para futuro carregamento"""
        
        if not os.path.exists(self.options.save_path):
            os.mkdir(self.options.save_path)

        with codecs.open(os.path.join(self.options.save_path, 'vocab_pt'), 'w', encoding='utf-8') as file_pt:
            for seq in self.data_pt:
                for word in seq:
                    file_pt.write('{} '.format(word))
            file_pt.write('\n{}\n'.format(self.options.vocabulary_size))
            for i in self.dict_pt.keys():
                file_pt.write("{}%@{}\n".format(i, self.dict_pt[i]))

        with codecs.open(os.path.join(self.options.save_path, 'vocab_en'), 'w', encoding='utf-8') as file_en:
            for seq in self.data_en:
                for word in seq:
                    file_en.write('{} '.format(word))
            file_en.write('\n{}\n'.format(self.options.vocabulary_size))
            for i in self.dict_en.keys():
                file_en.write("{}%@{}\n".format(i, self.dict_en[i]))
