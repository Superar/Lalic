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
data1 = list(zip(x1, y1))
data1 = [list(elem) for elem in data1]
class1 = [0 for _ in data1]

r2 = np.random.uniform(low=2.1, high=3.0, size=(101))
x2 = r2 * np.cos(theta)
y2 = r2 * np.sin(theta)
data2 = list(zip(x2,y2))
data2 = [list(elem) for elem in data2]
class2 = [1 for _ in data2]

data = np.asarray(data1 + data2)
classes = np.asarray(class1 + class2)

# Criação do modelo
model = Sequential()
model.add(Dense(3, activation='tanh', input_shape=(2,)))
model.add(Dense(1, activation='sigmoid'))

# Treinamento
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(data, classes, epochs=900)

# Plot
plt.figure(1)

# Cria uma matriz com os dados para o plot das boundaries
h = .02
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Modelo de predição
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array([1 if value >= 0.5 else 0 for value in Z]) # Discretiza os pontos, pois estavam em sigmoid
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap='cool') # Decision boundary
plt.scatter(x1, y1, color='b') # Pontos de dados classe 1
plt.scatter(x2, y2, color='r') #Pontos de dados classe 2
plt.savefig('circular_decision.png')
plt.show()
