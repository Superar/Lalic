import numpy as np
import codecs
import re
import os
from nltk.tokenize.regexp import RegexpTokenizer
from collections import Counter
from itertools import islice
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing import text


# TODO: Melhorar Tokenização com o NLTK
# Stemming: nltk.stem.RSLPStemmer()
# Formato:
# Token Token Token Raíz@@ Sufixo Token Token Token...


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
            text_pt = os.path.join(options.save_path, 'vocab_pt')
            self.data_pt, self.dict_pt, self.rev_dict_pt = self._load_data(text_pt)
            text_en = os.path.join(options.save_path, 'vocab_en')
            self.data_en, self.dict_en, self.rev_dict_en = self._load_data(text_en)

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

        tokenizer = RegexpTokenizer(r'\w+')

    # Criação de duas listas:
    # tokenized_text mantém as sentenças, porém cada sentença está tokenizada
    # text_words representa todas as palavras do texto tokenizadas, sem sentenças
        tokenized_text = list()
        text_words = list()
        for sentence in input_text:
            tokenized_sentence = tokenizer.tokenize(sentence.lower())
            for word in tokenized_sentence:
                text_words.append(word)
            tokenized_text.append(tokenized_sentence)

    # count mantém todas as vocabulary_size palavras mais comuns no texto
    # Estas palavras são utilizadas para criar um dicionário de palavra : índice (word_index)
    # O índice 0 não é utilizado pois será utilizado no padding
        count = [['UKN', -1]]
        count.extend(Counter(text_words).most_common(self.options.vocabulary_size -1))

        word_index = dict()
        for word, _ in count:
            word_index[word] = len(word_index) + 1

    # As sentenças são transformadas em sequências de índices,
    # levando em consideração palavras fora do vocabulário
        sequences = list()
        for sentence in tokenized_text:
            seq = list()
            for word in sentence:
                if word in word_index:
                    index = word_index[word]
                else:
                    index = word_index['UKN']
                seq.append(index)
            sequences.append(seq)

        index_word = dict(zip(word_index.values(), word_index.keys()))

        return sequences, word_index, index_word


    def _save_vocab(self):
        """Salva vocabulário para futuro carregamento em um arquivo:
        NUMERO DE SENTENÇAS
        SENTENÇAS (uma por linha)
        NÚMERO DE PALAVRAS NO VOCABULÁRIO
        DICIONÁRIO PALAVRA%@ÍNDICE"""

        if not os.path.exists(self.options.save_path):
            os.mkdir(self.options.save_path)

        with codecs.open(os.path.join(self.options.save_path, 'vocab_pt'), 'w', encoding='utf-8') as file_pt:
            file_pt.write('{}\n'.format(len(self.data_pt)))
            for seq in self.data_pt:
                for word in seq:
                    file_pt.write('{} '.format(word))
                file_pt.write('\n')
            file_pt.write('{}\n'.format(self.options.vocabulary_size))
            for i in self.dict_pt.keys():
                file_pt.write("{}%@{}\n".format(i, self.dict_pt[i]))

        with codecs.open(os.path.join(self.options.save_path, 'vocab_en'), 'w', encoding='utf-8') as file_en:
            file_en.write('{}\n'.format(len(self.data_en)))
            for seq in self.data_en:
                for word in seq:
                    file_en.write('{} '.format(word))
                file_en.write('\n')
            file_en.write('{}\n'.format(self.options.vocabulary_size))
            for i in self.dict_en.keys():
                file_en.write("{}%@{}\n".format(i, self.dict_en[i]))


    def _load_data(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError('Arquivo {} não encontrado'.format(path))
        else:
            with codecs.open(path, mode='r', encoding='utf-8') as _file:
                num_seq = int(_file.readline())
                seq_lines = list(islice(_file, num_seq))

                sequences = list()
                for seq in seq_lines:
                    sequences.append([int(x) for x in seq.split()])

                vocab_size = int(_file.readline())
                lines = list(islice(_file, vocab_size))

                word_index = dict()
                for line in lines:
                    (word, index) = line.strip().split('%@')
                    word_index[word] = int(index)

                rev_word_index = dict(zip(word_index.values(), word_index.keys()))

                return sequences, word_index, rev_word_index
