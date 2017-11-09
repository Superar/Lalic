# pylint: disable=C0111
from word_embeddings import WordEmbeddings
from gensim.models import Word2Vec
from gensim.models.word2vec import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


class Word2VecModel(WordEmbeddings):
    ''' Classe para operacoes com relacao ao modelo de word embeddings
        word2vec. As embeddings sao calculadas a partir de ``src_sentences``
    '''

    def __init__(self):
        WordEmbeddings.__init__(self)
        self.sentences = list()

    def train(self, lang, corpus_path, dim=100):
        ''' Treinamendo do modelo Word2Vec '''

        self.sentences = WordEmbeddings.process_sentences(lang, corpus_path)

        model_word2vec = Word2Vec(self.sentences, size=dim)
        # As embeddings serao salvas em ``modelo``
        self.model = model_word2vec.wv
        del model_word2vec

    def save(self, filename='model_word2vec.txt'):
        ''' Salva o modelo em ``filename`` '''

        self.model.save_word2vec_format(filename)

    def load(self, filename='model_word2vec.txt'):
        ''' Carrega modelo salvo em ``filename`` '''

        self.model = KeyedVectors.load_word2vec_format(filename)

    def plot(self, filename='word_embeddings_word2vec.png', num_points=500):
        ''' Traca o grafico com os vetores das palavras.
        Uso o modelo t-SNE do modulo sklearn para projetar os vetores
        para um espaco de 2 dimensoes. '''

        # Usa t-SNE para fazer as projecoes
        plot_data = [self.model[word]
                     for word in self.model.index2word[:num_points]]
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_data = tsne.fit_transform(plot_data)

        # Plot
        plt.figure(figsize=(18, 18))
        for i, word in enumerate(self.model.index2word[:num_points]):
            (x, y) = low_dim_data[i, :] # pylint: disable=C0103
            plt.scatter(x, y)
            plt.annotate(word,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='off points',
                         ha='right',
                         va='bottom')
        plt.savefig(filename)

    def plot_n_most_similar(self, word, num_neighbours=10,
                            filename='word2vec_most_similar.png'):
        ''' Traca o grafico com as n embeddings mais proximas de ``word``.
        Usa a distancia de cosseno como medida do quao proximas sao as palavras.
        '''

        plot_data = [self.model[word]]
        data_label = [word]

        for (_word, _) in self.model.most_similar(positive=[word],
                                                  topn=num_neighbours):
            plot_data.append(self.model[_word])
            data_label.append(_word)

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_data = tsne.fit_transform(plot_data)
        ref = low_dim_data[0]
        trans_low_dim_data = np.array([d - ref for d in low_dim_data])

        # Plot
        plt.figure(figsize=(18, 18))
        for i, _word in enumerate(data_label):
            (x, y) = trans_low_dim_data[i, :] # pylint: disable=C0103
            plt.scatter(x, y)
            plt.annotate(_word,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.savefig(filename)
