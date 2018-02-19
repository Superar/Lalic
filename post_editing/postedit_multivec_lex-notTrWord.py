import sys
sys.path.insert(0, '/home/marciolima/Documentos/Lalic/word_embeddings')
from read_blast import BlastReader
from multivec_embeddings import MultivecModel
from word2vec_embeddings import Word2VecModel

BLAST_FILE_PATH = '/home/marciolima/Dropbox/Marcio/Tarefas/1.Anotacao_erros_corpus_NMT/Blast/Entrada_Blast/test-a/FAPESP_NMT_test-a_truecased.txt'

blast_reader = BlastReader(BLAST_FILE_PATH)
errors = blast_reader.get_filtered_errors(['lex-notTrWord'])

multivec_model_en_pt = MultivecModel()
multivec_model_en_pt.load('/home/marciolima/Dropbox/Marcio/Tarefas/3.Geracao_APE_baseado_em_WE/MultiVec/Modelos-Marcio/model_multivec_fapesp100.txt')

word2vec_model_en = Word2VecModel()
word2vec_model_en.load('/home/marciolima/Dropbox/Marcio/Tarefas/3.Geracao_APE_baseado_em_WE/MultiVec/Modelos-Marcio/word2vec_model_bnc.txt')

for error in errors:
    line = error[0]
    print u' '.join(blast_reader.src_lines[line]).encode('utf-8')
    print u' '.join(blast_reader.ref_lines[line]).encode('utf-8')
    print u' '.join(blast_reader.sys_lines[line]).encode('utf-8')

    source_sentence = blast_reader.src_lines[line]
    sentence_to_correct = blast_reader.sys_lines[line]

    words_src = [blast_reader.src_lines[line][i] for i in error[1][0] if i > 0]
    words_sys = [blast_reader.sys_lines[line][i] for i in error[1][1] if i > 0]

    correcao_direta = multivec_model_en_pt.get_most_similar_tgt(words_sys[0].encode('utf-8'))[0].decode('utf-8')
    print u'mapeamento direto de sys: {} > {}'.format(' '.join(words_sys),
                                                      correcao_direta).encode('utf-8')

    closest_word = word2vec_model_en.get_most_similar_word(words_sys[0])
    correcao_indireta = multivec_model_en_pt.get_most_similar_tgt(closest_word.encode('utf-8'))[0].decode('utf-8')
    print u'mapeamento indireto de sys: {} > {} > {}'.format(' '.join(words_sys),
                                                           closest_word,
                                                           correcao_indireta).encode('utf-8')

    correcao_direta = multivec_model_en_pt.get_most_similar_tgt(words_src[0].encode('utf-8'))[0].decode('utf-8')
    print u'mapeamento direto de src: {} > {}'.format(' '.join(words_src),
                                                      correcao_direta).encode('utf-8')

    closest_word = word2vec_model_en.get_most_similar_word(words_src[0])
    correcao_indireta = multivec_model_en_pt.get_most_similar_tgt(closest_word.encode('utf-8'))[0].decode('utf-8')
    print u'mapeamento indireto de src: {} > {} > {}'.format(' '.join(words_src),
                                                             closest_word,
                                                             correcao_indireta).encode('utf-8')
    print
