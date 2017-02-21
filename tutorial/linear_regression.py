# coding=utf-8

"""Baseado em
https://github.com/hans/ipython-notebooks/blob/master/tf/TF%20tutorial.ipynb
Para rodar:
python linear_regression.py
tensorboar --logdit caminho/para/diretorio/temp"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tempfile


def main():
    # Cria dados aletórios aproximados por reta
    x_dados = np.linspace(-1, 1, 101)  # 101 números entre -1 e 1
    y_dados = x_dados * 2 + np.random.randn(*x_dados.shape) * 0.3  # 2x + e (aleatório)

    # Cria modelo para regressão
    # (None,) faz o placeholder ter 1 dimensão de tamanho variável
    # Placeholder é o que vai receber de entrada
    x = tf.placeholder(tf.float32, shape=(None,), name="x")
    y = tf.placeholder(tf.float32, shape=(None,), name="y")

    # Variable é o que vai estimar dentro do modelo
    w = tf.Variable(np.random.normal(), name="w")

    # Valores estimados pelo modelo são a multiplicação dos pesos W pela entrada x
    y_pred = tf.mul(w, x)

    # Inicia modelo
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Coloca dentro da sessão a fórmula para estimar (y_pred) e os placeholder que precisa (x)
    y_pred_dados = sess.run(y_pred, {x: x_dados})

    # Desenha a reta estimada com os pesos não otimizados
    plt.figure(1)
    plt.scatter(x_dados, y_dados)
    plt.plot(x_dados, y_pred_dados)

    # Cria uma função de custo para estimar os melhores pesos
    cost = tf.reduce_mean(tf.square(y_pred - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    # Criação do log para salvar os passos do treinamento e vizualizar no TensorBoard
    logdir = tempfile.mkdtemp()
    print(logdir)
    summary_op = tf.summary.scalar("cost", cost)
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)

    # Treinamento
    train_op = optimizer.minimize(cost)

    for t in range(30):
        cost_t, summary, _ = sess.run([cost, summary_op, train_op], {x: x_dados, y: y_dados})
        summary_writer.add_summary(summary, t)
        print cost_t.mean()

    # Gera dados novamente após o treinamento
    y_pred_dados = sess.run(y_pred, {x: x_dados})

    plt.figure(2)
    plt.scatter(x_dados, y_dados)
    plt.plot(x_dados, y_pred_dados)
    plt.show()

    # Envia dados para o TensorBoard
    summary_writer.flush()

if __name__ == '__main__':
    main()
