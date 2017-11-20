# pylint: disable=C0111
import string
import sys
from nltk.tokenize.moses import MosesTokenizer


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

        reload(sys)
        sys.setdefaultencoding('utf8')
        tokenizer = MosesTokenizer(lang=lang)
        # table = str.maketrans('', '', string.punctuation)

        with open(corpus_path, 'r') as _file:
            for sent in _file:
                # Lowercase e retira pontuacao
                proc_sent = sent.lower().translate(None, string.punctuation)
                tok_sent = tokenizer.tokenize(proc_sent, return_str=True)
                tokenized_sentences.append(tok_sent.split())

        return tokenized_sentences

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError

    def plot(self, filename, num_points):
        raise NotImplementedError
