from dataset import Dataset
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers.embeddings import Embedding


class Tradutor(object):
    """Modelo de tradutor neural"""

    def __init__(self, options):
        self.options = options
        self.dataset = Dataset(options)
        self.model = Sequential()
        self.build_model()

        print(self.model.summary())


    def build_model(self):
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
        print('Iniciando treinamento')
        self.model.fit(self.dataset.data_pt,
                       self.dataset.data_en,
                       epochs=self.options.iterations)

        print('Salvando modelo')
        self.model.save('modelo/model.h5', overwrite=True)

        print('Avaliando modelo')
        self.model.reset_states()
        scores = self.model.evaluate(dataset.data_pt, dataset.data_en)
        print(scores)