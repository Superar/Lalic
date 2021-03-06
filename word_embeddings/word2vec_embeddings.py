# pylint: disable=C0111
from word_embeddings import WordEmbeddings
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt


class Word2VecModel(WordEmbeddings):
    ''' Classe para operacoes com relacao ao modelo de word embeddings
        word2vec. As embeddings sao calculadas a partir de ``src_sentences``
    '''

    def __init__(self, skipgram=False):
        WordEmbeddings.__init__(self)
        self.sentences = list()
        self.sg = skipgram

    def train(self, lang, corpus_path, dim=100):
        ''' Treinamendo do modelo Word2Vec '''

        self.sentences = WordEmbeddings.process_sentences(lang, corpus_path)

        model_word2vec = Word2Vec(self.sentences, size=dim, sg=self.sg)
        self.model = model_word2vec
        del model_word2vec

    def save(self, filename='model_word2vec.txt'):
        ''' Salva o modelo em ``filename`` '''

        self.model.save(filename)

    def load(self, filename='model_word2vec.txt'):
        ''' Carrega modelo salvo em ``filename`` '''

        self.model = Word2Vec.load(filename)

    def plot(self, filename='word_embeddings_word2vec.png', num_points=500, figsize=(18, 18)):
        ''' Traca o grafico com os vetores das palavras.
        Uso o modelo t-SNE do modulo sklearn para projetar os vetores
        para um espaco de 2 dimensoes. '''

        # Usa t-SNE para fazer as projecoes
        data_labels = [word.encode('utf-8')
                       for word in self.model.wv.index2word[:num_points]]
        plot_vectors = [self.model.wv[word]
                        for word in self.model.wv.index2word[:num_points]]
        data = dict(zip(data_labels, plot_vectors))

        # Plot
        fig = plt.figure(figsize=figsize)
        graphics = fig.add_subplot(111)
        self._scatter_data(graphics, data)
        fig.savefig(filename)

    def plot_n_most_similar(self, word, num_neighbours=10,
                            filename='word2vec_most_similar.png',
                            figsize=(18, 18)):
        ''' Traca o grafico com as n embeddings mais proximas de ``word``.
        Usa a distancia de cosseno como medida do quao proximas sao as palavras.
        '''

        word = word.decode('utf8')

        plot_vectors = [self.model.wv[word]]
        data_label = [word.encode('utf-8')]

        for (_word, _) in self.model.wv.most_similar(positive=[word],
                                                     topn=num_neighbours):
            plot_vectors.append(self.model.wv[_word])
            data_label.append(_word.encode('utf-8'))

        ref = plot_vectors[0]
        trans_data = [d - ref for d in plot_vectors]
        data = dict(zip(data_label, trans_data))

        fig = plt.figure(figsize=figsize)
        graphics = fig.add_subplot(111)
        self._scatter_data(graphics, data)
        fig.savefig(filename)

    def get_most_similar_word(self, word):
        try:
            closest_word = self.model.wv.most_similar(positive=[word],
                                                      topn=1)[0][0]
        except KeyError:
            closest_word = '***'

        return closest_word
