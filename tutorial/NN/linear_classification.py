import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


# Cria os dados aleatórios
x = np.linspace(-1, 1, 200)
y = 2 * np.random.uniform(low=-1.0, high=1.0, size=(200)) - 1

# Cria uma decision boundary arbitrária
boundary = 2 * x - 1

# Separa as classes entre: abaixo e acima da curva
under_curve = y < boundary
classes = [1 if under else 2 for under in under_curve]
colors = ['b' if under else 'r' for under in under_curve]

data = list(zip(x, y))

# Treinamento usando SVM Linear
model = LinearSVC()
model.fit(data, classes)

# Criação da linha da decision boundary aprendida
w = model.coef_[0]
b = model.intercept_[0]
y_pred_depois = (-w[0] / w[1]) * x - (b / w[1])

# Plot
plt.figure(1)
plt.scatter(x, y, color=colors)
plt.plot(x, boundary, 'g')
plt.plot(x, y_pred_depois, 'c')
plt.show()
