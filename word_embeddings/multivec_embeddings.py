# pylint: disable=C0111
# pylint: disable=R0913
from tempfile import NamedTemporaryFile
from word_embeddings import WordEmbeddings
from multivec import BilingualModel # pylint: disable=E0611

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
            src_file.write(' '.join(sentence))
            src_file.write('\n')

        for sentence in self.sentences_tgt:
            tgt_file.write(' '.join(sentence))
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

    def plot(self, filename='word_embeddings_multivec.png', num_points=500):
        pass
