import os
import numpy as np
from dataset import Dataset
from keras.models import load_model
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers.embeddings import Embedding


# TODO: Calcular precisão com NLTK.bleu
# TODO: Melhorar modelo. Opções:
# Adicionais mais camadas RNN
# Adicionar Attention


class Tradutor(object):
    """Modelo de tradutor neural.

    O modelo consiste de:

    dataset - Conjunto de dados para treinamento

    model - Modelo Keras para a rede neural
    """

    def __init__(self, options):
        self.options = options
        self.dataset = Dataset(options)

        model_path = os.path.join(options.save_path, 'modelo.h5')
        if options.load:
            if not os.path.isfile(model_path):
                raise FileNotFoundError('Arquivo {} não encontrado'.format(model_path))
            self.model = load_model(model_path)
        else:
            self.model = Sequential()
            self._build_model()

        print(self.model.summary())


    def _build_model(self):
        """Constrói o modelo de rede neural."""

        hidden_size = self.options.hidden_size

        self.model.add(Embedding(self.options.vocabulary_size,
                            self.options.embedding_size,
                            input_length=self.options.sequence_length,
                            mask_zero=True))
        # self.model.add(GRU(hidden_size,
        #               input_shape=(self.options.sequence_length, self.options.embedding_size),
        #               return_sequences=True))
        # self.model.add(GRU(hidden_size,
        #               input_shape=(None, hidden_size),
        #               return_sequences=True))
        # self.model.add(GRU(hidden_size,
        #                    input_shape=(None, hidden_size),
        #                    return_sequences=True))
        self.model.add(GRU(hidden_size,
                           input_shape=(None, hidden_size)))
        self.model.add(Dense(self.options.hidden_size,
                        input_shape=(None, hidden_size)))
        self.model.add(Activation('relu'))
        self.model.add(RepeatVector(self.options.sequence_length,
                               input_shape=(None, self.options.hidden_size)))
        # self.model.add(GRU(self.options.hidden_size,
        #               input_shape=(None, self.options.hidden_size),
        #               return_sequences=True))
        # self.model.add(GRU(self.options.hidden_size,
        #               input_shape=(None, self.options.hidden_size),
        #               return_sequences=True))
        # self.model.add(GRU(self.options.hidden_size,
        #               input_shape=(None, self.options.hidden_size),
        #               return_sequences=True))
        self.model.add(GRU(self.options.hidden_size,
                      input_shape=(None, self.options.hidden_size),
                      return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.options.vocabulary_size),
                                       input_shape=(self.options.sequence_length,1)))
        self.model.add(Activation('softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    def train(self):
        """Realiza o treinamento da rede com os dados em dataset."""

        target = np.empty((self.options.sequence_length, self.options.vocabulary_size + 1))
        for seq in self.dataset.data_en:
            cat_seq = np.zeros((self.options.sequence_length, self.options.vocabulary_size + 1),
                               dtype=np.bool)
            cat_seq[np.arange(self.options.sequence_length), seq] = 1
            np.vstack((target, cat_seq))


        print('Iniciando treinamento')
        self.model.fit(self.dataset.data_pt,
                       self.dataset.data_en,
                       epochs=self.options.iterations)


    def save(self):
        """Salva o modelo em um arquivo SAVE_PATH/modelo.h5"""
        print('Salvando modelo')
        self.model.save(os.path.join(self.options.save_path, 'modelo.h5'), overwrite=True)


    def evaluate(self):
        """Imprime na tela os scores do modelo sendo: [loss, accuracy]"""
        print('Avaliando modelo')
        self.model.reset_states()
        scores = self.model.evaluate(self.dataset.data_pt, self.dataset.data_en)
        print(scores)
