# -*- coding: utf-8 -*-

import codecs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-src', '--source_path',
                    help='Caminho para texto no idioma fonte', type=str, default=None, required=True)
parser.add_argument('-ref', '--reference_path',
                    help='Caminho para texto de referência no idioma alvo', type=str, default=None, required=True)
parser.add_argument('-out', '--output_path',
                    help='Caminho para saída do tradutor automático no idioma alvo', type=str, default=None, required=True)
parser.add_argument('-blast', '--blast_path',
                    help='Caminho onde deve ser salvo o arquivo BLAST', type=str, default=None, required=True)
FLAGS = parser.parse_args()

src_file_path = FLAGS.source_path
ref_file_path = FLAGS.reference_path
out_file_path = FLAGS.output_path
blast_file_path = FLAGS.blast_path


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
