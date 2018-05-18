# pylint: disable=C0111
# pylint: disable=R0913
from tempfile import NamedTemporaryFile
from word_embeddings import WordEmbeddings
from multivec import BilingualModel # pylint: disable=E0611
import matplotlib.pyplot as plt

class MultivecModel(WordEmbeddings):

    def __init__(self):
        WordEmbeddings.__init__(self)
        self.sentences_src = list()
        self.sentences_tgt = list()

    def train(self, lang_src, lang_tgt,
              corpus_path_src, corpus_path_tgt,
              dim=100):

        self.sentences_src = WordEmbeddings.process_sentences(lang_src, corpus_path_src)
        self.sentences_tgt = WordEmbeddings.process_sentences(lang_tgt, corpus_path_tgt)

        src_file = NamedTemporaryFile(mode='w')
        tgt_file = NamedTemporaryFile(mode='w')

        for sentence in self.sentences_src:
            src_file.write(' '.join(sentence).encode('utf8'))
            src_file.write('\n')

        for sentence in self.sentences_tgt:
            tgt_file.write(' '.join(sentence).encode('utf8'))
            tgt_file.write('\n')

        self.model = BilingualModel(dimension=dim, threads=16)
        self.model.train(src_file.name,
                         tgt_file.name)

        src_file.close()
        tgt_file.close()

    def save(self, filename='model_multivec.txt'):
        self.model.save(filename)

    def load(self, filename='model_multivec.txt'):
        self.model = BilingualModel(name=filename)

    def plot(self, filename='word_embeddings_multivec', num_points=500, figsize=(18, 18)):
        fig_src = plt.figure(figsize=figsize)
        graphic_src = fig_src.add_subplot(111)
        fig_tgt = plt.figure(figsize=figsize)
        graphic_tgt = fig_tgt.add_subplot(111)

        data_labels_src = self.model.src_model.get_vocabulary()[:num_points]
        plot_vectors_src = [self.model.src_model.word_vec(w) for w in data_labels_src]
        data_src = dict(zip(data_labels_src, plot_vectors_src))
        self._scatter_data(graphic_src, data_src)

        data_labels_tgt = self.model.trg_model.get_vocabulary()[:num_points]
        plot_vectors_tgt = [self.model.trg_model.word_vec(w) for w in data_labels_tgt]
        data_tgt = dict(zip(data_labels_tgt, plot_vectors_tgt))
        self._scatter_data(graphic_tgt, data_tgt)

        fig_src.savefig(filename + 'src.png')
        fig_tgt.savefig(filename + 'tgt.png')

    def get_most_similar_src(self, word):
        try:
            similar_words = [w[0] for w in self.model.src_closest(word)]
        except RuntimeError:
            similar_words = ['***']
        return similar_words

    def get_most_similar_tgt(self, word):
        try:
            similar_words = [w[0] for w in self.model.trg_closest(word)]
        except RuntimeError:
            similar_words = ['***']
        return similar_words
