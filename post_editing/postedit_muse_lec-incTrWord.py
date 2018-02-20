# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '/home/marciolima/Documentos/Lalic/word_embeddings')
from read_blast import BlastReader
from read_muse_embeds import load_embeddings, closest_words

BLAST_FILE_PATH = '/home/marcio/Dropbox/Marcio/Tarefas/1.Anotacao_erros_corpus_NMT/Blast/Entrada_Blast/test-a/FAPESP_NMT_test-a_truecased.txt'
MUSE_EN_FILE_PATH = '../word_embeddings/models/7evwxf1kof/vectors-en.txt'
MUSE_PT_FILE_PATH = '../word_embeddings/models/7evwxf1kof/vectors-pt.txt'

blast_reader = BlastReader(BLAST_FILE_PATH)
errors = blast_reader.get_filtered_errors(['lex-incTrWord'])

emb_en, emb_pt = load_embeddings(MUSE_EN_FILE_PATH, MUSE_PT_FILE_PATH)

for error in errors:
    line = error[0]
    print(' '.join(blast_reader.src_lines[line]))
    print(' '.join(blast_reader.ref_lines[line]))
    print(' '.join(blast_reader.sys_lines[line]))

    sentence_to_correct = blast_reader.src_lines[line]
    for i in error[1][0]:
        if i > 0:
            corrected_word = closest_words(sentence_to_correct[i], emb_en, emb_pt)[0][0]
            print('corrected: {} > {}'.format(sentence_to_correct[i],
                                               corrected_word))
            sentence_to_correct[i] = corrected_word
    print(' '.join(sentence_to_correct))
    print('\n')
