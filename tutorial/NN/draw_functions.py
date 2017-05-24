import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)


x = np.linspace(-5, 5, 100)

plt.figure('Sigmoid')
plt.plot(x, sigmoid(x))

plt.figure('Tanh')
plt.plot(x, np.tanh(x))

plt.figure('ReLU')
plt.plot(x, relu(x))
plt.show()
