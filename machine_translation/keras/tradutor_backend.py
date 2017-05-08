import os
from dataset import Dataset
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers.embeddings import Embedding


class Tradutor(object):
    """Modelo de tradutor neural.

    O modelo consiste de:

    dataset - Conjunto de dados para treinamento

    model - Modelo Keras para a rede neural
    """

    def __init__(self, options):
        self.options = options
        self.dataset = Dataset(options)
        self.model = Sequential()
        self._build_model()

        print(self.model.summary())


    def _build_model(self):
        """Constrói o modelo de rede neural."""

        hidden_size = 512

        self.model.add(Embedding(self.options.vocabulary_size,
                            self.options.embedding_size,
                            input_length=self.options.sequence_length,
                            mask_zero=True))
        self.model.add(GRU(hidden_size,
                      input_shape=(self.options.sequence_length, self.options.embedding_size)))
        self.model.add(Dense(self.options.hidden_size,
                        input_shape=(None, self.options.hidden_size)))
        self.model.add(Activation('relu'))
        self.model.add(RepeatVector(self.options.sequence_length,
                               input_shape=(None, self.options.hidden_size)))
        self.model.add(GRU(self.options.hidden_size,
                      input_shape=(None, self.options.hidden_size),
                      return_sequences=True))
        self.model.add(TimeDistributed(Dense(1, activation='softmax'),
                                  input_shape=(self.options.sequence_length,1)))
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


    def train(self):
        """Realiza o treinamento da rede com os dados em dataset."""

        print('Iniciando treinamento')
        self.model.fit(self.dataset.data_pt,
                       self.dataset.data_en,
                       epochs=self.options.iterations)


    def save(self):
        print('Salvando modelo')
        self.model.save(os.path.join(self.options.save_path, 'modelo.h5'), overwrite=True)


    def evaluate(self):
        print('Avaliando modelo')
        self.model.reset_states()
        scores = self.model.evaluate(self.dataset.data_pt, self.dataset.data_en)
        print(scores)
