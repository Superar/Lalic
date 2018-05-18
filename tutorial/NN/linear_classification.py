import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation


# Cria os dados aleatórios
x = np.linspace(-1, 1, 200)
y = 2 * np.random.uniform(low=-1.0, high=1.0, size=(200)) - 1

# Cria uma decision boundary arbitrária
boundary = 2 * x - 1

# Separa as classes entre: abaixo e acima da curva
under_curve = y < boundary
classes = [0 if under else 1 for under in under_curve]
colors = ['b' if under else 'r' for under in under_curve]

data = list(zip(x, y))
data = [list(elem) for elem in data]

# Treinamento usando perceptron
model = Sequential()
model.add(Dense(1, activation='sigmoid', input_shape=(2,)))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(data, classes, epochs=1000)

w = model.get_weights()[0]
b = model.get_weights()[1]

y_pred_depois = x * (-w[0] / w[1]) - (b / w[1])

# Plot
plt.figure(1)
plt.scatter(x, y, color=colors)
plt.plot(x, boundary, 'g')
plt.plot(x, y_pred_depois, 'c')
plt.show()
