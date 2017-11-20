# pylint: disable=C0111
# pylint: disable=R0913
from word_embeddings import WordEmbeddings
from multivec import BilingualModel

class MultivecModel(WordEmbeddings):

    def __init__(self):
        WordEmbeddings.__init__(self)
        self.sentences_src = list()
        self.sentences_tgt = list()

    def train(self, lang_src, lang_tgt,
              corpus_path_src, corpus_path_tgt,
              dim=100):

        # self.sentences_src = WordEmbeddings.process_sentences(lang_src, corpus_path_src)
        # self.sentences_tgt = WordEmbeddings.process_sentences(lang_tgt, corpus_path_tgt)

        # src_file = open('src_file.' + lang_src, 'w')
        # tgt_file = open('tgt_file.' + lang_tgt, 'w')

        # for sentence in self.sentences_src:
        #     src_file.write(' '.join(sentence))
        #     src_file.write('\n')

        # for sentence in self.sentences_tgt:
        #     tgt_file.write(' '.join(sentence))
        #     tgt_file.write('\n')

        # src_file.close()
        # tgt_file.close()

        self.model = BilingualModel(dimension=dim, threads=16)
        self.model.train('src_file.' + lang_src,
                         'tgt_file.' + lang_tgt)

    def save(self, filename='model_multivec.txt'):
        pass

    def load(self, filename='model_multivec.txt'):
        pass

    def plot(self, filename='word_embeddings_multivec.png', num_points=500):
        pass
