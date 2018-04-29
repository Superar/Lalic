# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# 60k imagens de treinamento e 10k de teste
# Imagens de 28 x 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Transformando imagens em vetores únicos
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# Normalização
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Transformando números em vetores binários
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#Criação do modelo
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', input_shape=(None, 512)))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax', input_shape=(None, 512)))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])

# Treinamento
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# Avaliação do modelo
score_depois = model.evaluate(x_test, y_test)

print("\nPrecisão: {}".format(score_depois[1]))

# Plot de gráfico
plt.figure(1)
plt.plot(history.history['acc'])
plt.show()
