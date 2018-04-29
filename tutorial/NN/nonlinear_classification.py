import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from scipy.stats import norm


# Cria os dados aleatórios
x = np.linspace(-1, 1, 200)
y = np.random.uniform(low=-1.0, high=1.0, size=(200))
data = [list(elem) for elem in zip(x, y)]
data = np.array(data)

# Cria uma decision boundary arbitrária
# boundary = np.tanh(x, out=None)
boundary = norm.pdf(x, loc=0, scale=0.3)

# Separa as classes entre: abaixo e acima da curva
under_curve = y < boundary
classes = [0 if under else 1 for under in under_curve]
colors = ['b' if under else 'r' for under in under_curve]

# Criação do modelo
model = Sequential()
model.add(Dense(3, activation='tanh', input_shape=(2,)))
model.add(Dense(1, activation='sigmoid'))

# Treinamento
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(data, classes, epochs=800)

# Cria matriz com valores para plotar
h = .02
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Valores de predição
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array([1 if value >= 0.5 else 0 for value in Z]) # Discretiza os pontos, pois estavam em sigmoid
Z = Z.reshape(xx.shape)

# Plot
plt.figure('Modelo treinado')
plt.contourf(xx, yy, Z, cmap='cool') # Decision boundary
plt.scatter(x, y, color=colors)

plt.figure('Dados não-separáveis linearmente')
plt.scatter(x, y, color=colors)
plt.show()
