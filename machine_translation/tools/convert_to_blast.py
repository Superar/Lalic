# -*- coding: utf-8 -*-

import codecs

src_file_path = '../Corpus_FAPESP_v2/corpus_teste/pt-en/fapesp-v2.pt-en.test-a.en.atok'
ref_file_path = '../Corpus_FAPESP_v2/corpus_teste/pt-en/fapesp-v2.pt-en.test-a.pt.atok'
out_file_path = '../Corpus_FAPESP_v2/corpus_teste/pt-en/2/fapesp-v2.pt-en.test-a.truecased'
blast_file_path = '../Corpus_FAPESP_v2/corpus_teste/pt-en/2/FAPESP_NMT_test-a.txt'


src_file = codecs.open(src_file_path, encoding='utf-8')
ref_file = codecs.open(ref_file_path, encoding='utf-8')
out_file = codecs.open(out_file_path, encoding='utf-8')
blast_file = codecs.open(blast_file_path, encoding='utf-8', mode='w')

blast_file.write('#Sentencetypes src ref sys\n#catfile lalic-catsv2\n')

for linha in src_file:
    blast_file.write(linha)
    blast_file.write(ref_file.readline())
    blast_file.write(out_file.readline())
    blast_file.write('\n')
    blast_file.write('\n')
