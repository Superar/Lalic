# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model


# Cria dados aletórios aproximados por reta
x_dados = np.linspace(-1, 1, 101)  # 101 números entre -1 e 1
y_dados = x_dados * 2 + np.random.randn(*x_dados.shape) * 0.3  # 2x + e (aleatório)

# Criação do modelo
model = Sequential()
# Adição de uma camada Dense
model.add(Dense(1, input_shape=(1,)))

# Predição antes do treinamento
y_pred_antes = model.predict(x_dados)

# Treinamento com 70 iterações
model.compile(optimizer='sgd', loss='mse')

model.fit(x_dados, y_dados, epochs=70)

# Predição após treinamento
y_pred_depois = model.predict(x_dados)

# Plot
plt.figure(1)
plt.scatter(x_dados, y_dados)
plt.plot(x_dados, y_pred_antes, color='r')
plt.plot(x_dados, y_pred_depois, color='g')
plt.show()
