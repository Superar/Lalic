from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.manifold import TSNE
from nltk.tokenize.moses import MosesTokenizer
import matplotlib.pyplot as plt
import os
import subprocess
import string


class WordEmbeddings(object):
    ''' Techniques for computing of word embeddings

    sentences - Stores an list of lists.
                Each sublist is a sentence for training '''

    def __init__(self):
        self.sentences = []

    def process_sentences(self, corpus_path, tokenizer_path):
        ''' Data preprocessing. Tokenization.

        corpus_path - Path to the text
        tokenizer_path - Path to the perl tokenizer from Moses '''

        # Tokenizer must exist
        if not os.path.isfile(tokenizer_path):
            raise(FileNotFoundError('[Errno ' + str(os.errno.ENOENT) + '] ' +
                                    'No such file or directory: ' +
                                    '\'' + tokenizer_path + '\''))
        # Tokenization
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

        # Lowercase and remove punctuation
        # format [['first', 'sentence'], ['second', 'sentence']]

        table = str.maketrans('', '', string.punctuation)
        for sentence in raw_sentences:
            new_sentence = sentence.decode('utf-8').lower().translate(table)
            self.sentences.append(new_sentence.split())

    def train_word2vec(self):
        ''' Training of the word2vec model '''

        model = Word2Vec(self.sentences)
        # The embeddings are stored at word2vec_model
        self.word2vec_model = model.wv
        del model

    def save_word2vec(self, filename='word2vec_model.txt'):
        ''' Save model '''

        self.word2vec_model.save_word2vec_format(filename)

    def load_word2vec(self, filename='word2vec_model.txt'):
        ''' Load model '''

        self.word2vec_model = KeyedVectors.load_word2vec_format(filename)

    def plot(self, filename='word_embeddings.png', num_points=500):
        ''' Plot the embeddings.
        This uses the t-SNE model from the sklearn model to project
        the vectors to a 2-dimension space. '''

        # Use t-SNE to do the projections
        plot_data = [self.word2vec_model[word]
                     for word in self.word2vec_model.index2word[:num_points]]
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_data = tsne.fit_transform(plot_data)

        # Plot
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

    def plot_n_most_similar(self, word, n=10,
                            filename='word2vec_most_similar.png'):
        ''' Plot the n most close embeddings from ``word``.
        This uses the cossine distance as a measure of how close the words are. '''

        plot_data = [self.word2vec_model[word]]
        data_label = [word]

        for (word, _) in self.word2vec_model.most_similar(positive=[word],
                                                          topn=n):
            plot_data.append(self.word2vec_model[word])
            data_label.append(word)

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_data = tsne.fit_transform(plot_data)

        # Plot
        plt.figure(figsize=(18, 18))
        for i, word in enumerate(data_label):
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
print(MosesTokenizer(lang='pt').tokenize('Oi, tudo bem? Estou testando esse ``tokenizador\'\''))
