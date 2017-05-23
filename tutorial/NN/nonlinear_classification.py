import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation


# Cria os dados aleatórios
x = np.linspace(-100, 100, 200)
y = 2 * np.random.uniform(low=-1.0, high=1.0, size=(200)) - 1
data = [list(elem) for elem in zip(x, y)]
data = np.asarray(data)

# Cria uma decision boundary arbitrária
boundary = np.tanh(x, out=None)

# Separa as classes entre: abaixo e acima da curva
under_curve = y < boundary
classes = [0 if under else 1 for under in under_curve]
colors = ['b' if under else 'r' for under in under_curve]

# Criação do modelo
model = Sequential()
model.add(Dense(1, input_shape=(2,)))
model.add(Activation('tanh'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(data, classes, epochs=100)

class_pred = model.predict(data)

# Plot
plt.figure(1)
plt.scatter(x, y, color=colors)
plt.plot(x, boundary, 'g')
plt.plot(x, class_pred, 'r')
plt.savefig('tanh_data.png')
plt.show()
