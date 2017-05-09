"""
Tutorial reproduzido de
http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
"""

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

numpy.random.seed(seed=7)

# Cria o dataset
vocab_size = 5000
(in_train, out_train), (in_test, out_test) = imdb.load_data(num_words=vocab_size)

# Deixa as sequências todas com o mesmo tamanho, para não dar erro de dimensão
# durante a execução. O sistema irá aprender que 0 não carrega informação
max_seq_length = 500
in_train = sequence.pad_sequences(in_train, maxlen=max_seq_length)
in_test = sequence.pad_sequences(in_test, maxlen=max_seq_length)

# Criação do modelo
embedding_dim = 32
model = Sequential()
# Word ebeddings
model.add(Embedding(vocab_size, embedding_dim, input_length=max_seq_length))
# Modelo de LSTM com saída de dimensão 100
model.add(LSTM(100))
# Modelo calcula uma sigmoid para gerar duas saídas: 0 ou 1
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

print(model.summary())
model.fit(in_train, out_train,
          batch_size=64, epochs=3,
          validation_data=(in_test, out_test))

# Calculando a precisão do modelo
scores = model.evaluate(in_test, out_test, verbose=0)
print("Precisão {}%".format(scores[1]*100))
