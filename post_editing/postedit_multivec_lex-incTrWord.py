import sys
sys.path.insert(0, '/home/marciolima/Documentos/Lalic/word_embeddings')
from read_blast import BlastReader
from word2vec_embeddings import Word2VecModel

BLAST_FILE_PATH = '/home/marciolima/Dropbox/Marcio/Tarefas/1.Anotacao_erros_corpus_NMT/Blast/Entrada_Blast/test-a/FAPESP_NMT_test-a_truecased.txt'

blast_reader = BlastReader(BLAST_FILE_PATH)
errors = blast_reader.get_filtered_errors(['lex-incTrWord'])

word2vec_model_pt = Word2VecModel()
word2vec_model_pt.load('modelo_word2vec_pt_100.txt')

for error in errors:
    line = error[0]
    print u' '.join(blast_reader.src_lines[line]).encode('utf-8')
    print u' '.join(blast_reader.ref_lines[line]).encode('utf-8')
    print u' '.join(blast_reader.sys_lines[line]).encode('utf-8')

    sentence_to_correct = blast_reader.sys_lines[line]
    for i in error[1][1]:
        if i > 0:
            corrected_word = word2vec_model_pt.get_most_similar_word(sentence_to_correct[i])
            print u'corrected: {} > {}'.format(sentence_to_correct[i],
                                               corrected_word).encode('utf-8')
            sentence_to_correct[i] = corrected_word
    print u' '.join(sentence_to_correct).encode('utf-8')
    print
