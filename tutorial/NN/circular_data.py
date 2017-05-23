# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model


# Cria um círculo em coordenadas polares
theta = np.linspace(0, 2*np.pi, 101) #101 pontos entre [0, 2pi]

# Cria os pontos aleatórios
r1 = np.random.uniform(low=0.0, high=2.0, size=(101)) # Escolhe raios aletaórios
x1 =  r1 * np.cos(theta) # Multiplica raios sobre os valores do círculo para gerar os pontos
y1 = r1 * np.sin(theta)

r2 = np.random.uniform(low=2.1, high=3.0, size=(101))
x2 = r2 * np.cos(theta)
y2 = r2 * np.sin(theta)

x = np.append(x1, x2)
y = np.append(y1, y2)

# Criação do modelo
model = Sequential()
# Adição de uma camada Dense
model.add(Dense(1, input_shape=(1,)))

# Predição antes do treinamento
y_pred_antes = model.predict(x)

# Treinamento com 70 iterações
model.compile(optimizer='sgd', loss='mse')

model.fit(x, y, epochs=70)

# Predição após treinamento
y_pred_depois = model.predict(x)

# Plot
plt.figure(1)
plt.scatter(x1, y1, color='b')
plt.scatter(x2, y2, color='r')
plt.plot(x, y_pred_antes, color='r')
plt.plot(x, y_pred_depois, color='g')
plt.show()
