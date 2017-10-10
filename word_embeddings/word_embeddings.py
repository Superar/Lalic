from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import subprocess


class WordEmbeddings(object):
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.sentences = []

    def process_sentences(self, tokenizer_path):
        with open(self.corpus_path, 'rb') as _file:
            tokenizer_process = subprocess.Popen(['perl', tokenizer_path, '-a',
                                                  '-no-escape', '-l pt', '-q'],
                                                 stdin=_file,
                                                 stdout=subprocess.PIPE)
            raw_sentences = tokenizer_process.stdout.readlines()
            tokenizer_process.stdout.close()

        for sentence in raw_sentences:
            self.sentences.append(sentence.decode('utf-8').split())

    def train_word2vec(self):
        model = Word2Vec(self.sentences)
        self.word2vec_model = model.wv
        del model

    def save_Word2Vec(self, filename):
        self.word2vec_model.save_word2vec_format(filename)

    def load_word2vec(self, filename):
        self.word2vec_model = KeyedVectors.load_word2vec_format(filename)


embedder = WordEmbeddings('../Corpus_FAPESP_pt-en_bitexts/fapesp-bitexts.pt-en.pt')
embedder.process_sentences('../machine_translation/OpenNMT/tokenizer.perl')
embedder.train_word2vec()
embedder.save_Word2Vec('word2vec_model.txt')
