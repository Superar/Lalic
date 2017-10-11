from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import subprocess


class WordEmbeddings(object):
    def __init__(self):
        self.sentences = []

    def process_sentences(self, corpus_path, tokenizer_path):
        if not os.path.isfile(tokenizer_path):
            raise(FileNotFoundError('[Errno ' + str(os.errno.ENOENT) + '] ' +
                                    'No such file or directory: ' +
                                    '\'' + tokenizer_path + '\''))
        with open(corpus_path, 'rb') as _file:
            tokenizer_process = subprocess.Popen(['perl', tokenizer_path,
                                                  '-a',
                                                  '-no-escape',
                                                  '-l', 'pt',
                                                  '-q'],
                                                 stdin=_file,
                                                 stdout=subprocess.PIPE)
            raw_sentences = tokenizer_process.stdout.readlines()
            tokenizer_process.stdout.close()

        for sentence in raw_sentences:
            self.sentences.append(sentence.decode('utf-8').lower().split())

    def train_word2vec(self):
        model = Word2Vec(self.sentences)
        self.word2vec_model = model.wv
        del model

    def save_word2vec(self, filename='word2vec_model.txt'):
        self.word2vec_model.save_word2vec_format(filename)

    def load_word2vec(self, filename='word2vec_model.txt'):
        self.word2vec_model = KeyedVectors.load_word2vec_format(filename)

    def plot(self, filename='word_embeddings.png', num_points=500):
        plot_data = [self.word2vec_model[word]
                     for word in self.word2vec_model.index2word[:num_points]]
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_data = tsne.fit_transform(plot_data)

        plt.figure(figsize=(18, 18))
        for i, word in enumerate(self.word2vec_model.index2word[:num_points]):
            x, y = low_dim_data[i, :]
            plt.scatter(x, y)
            plt.annotate(word,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.savefig(filename)


# embedder = WordEmbeddings()
# embedder.process_sentences('../Corpus_FAPESP_pt-en_bitexts/fapesp-bitexts.pt-en.pt',
#                            'tokenizer.perl')
# embedder.train_word2vec()
# embedder.save_word2vec()
embedder = WordEmbeddings()
embedder.load_word2vec()
embedder.plot()
