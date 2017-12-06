# pylint: disable=C0111
import string
from nltk.tokenize.moses import MosesTokenizer
from sklearn.manifold import TSNE


class WordEmbeddings(object):
    ''' Tecnicas para computar word embeddings

    ``model`` - Modelo treinado
    '''

    def __init__(self):
        self.model = None

    @staticmethod
    def process_sentences(lang, corpus_path):
        ''' Pre-processamento dos dados. Toquenizacao.

        ``lang`` - Idioma para a tokenizacao.
        ``corpus_path`` - Caminho para o texto. Uma sentenca por linha.

        As sentencas tokenizadas sao retornadas no formato:
        [['primeira', 'sentenca'], ['segunda', 'sentenca']]
        '''

        tokenized_sentences = list()

        tokenizer = MosesTokenizer(lang=lang)

        with open(corpus_path, 'r') as _file:
            for sent in _file:
                sent = sent.decode('utf8')
                # Lowercase e retira pontuacao
                proc_sent = ''.join(c for c in sent.lower() if c not in string.punctuation)
                tok_sent = tokenizer.tokenize(proc_sent, return_str=True)
                tokenized_sentences.append(tok_sent.split())

        return tokenized_sentences

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError

    def plot(self, filename, num_points, figsize):
        raise NotImplementedError

    @staticmethod
    def _scatter_data(fig, data):
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_data = tsne.fit_transform(data.values())

        # Plot
        for i, word in enumerate(data):
            (x, y) = low_dim_data[i, :] # pylint: disable=C0103
            fig.scatter(x, y)
            fig.annotate(word.decode('utf-8'),
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
