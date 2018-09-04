#!/usr/bin/env python3
import argparse
import progressbar
import os
from gensim.models import Word2Vec

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i',
                    help='Caminho para o arquivo anotado em sentenças com palavra_TAG',
                    required=True,
                    type=str)
parser.add_argument('--output', '-o',
                    help='Caminho para o arquivo em que o modelo será salvo',
                    required=False,
                    type=str)

FLAGS = parser.parse_args()

output_path = FLAGS.output if FLAGS.output else os.path.splitext(FLAGS.input)[
    0] + '.model'

print('Lendo arquivo')
sentences = list()
with open(FLAGS.input) as _file:
    lines = _file.readlines()
    for line in progressbar.progressbar(lines):
        s_line = line.split()
        sentences.append(s_line)

print('Treinando modelo')
model = Word2Vec(sentences)
model.wv.save_word2vec_format(output_path)
print('Finalizado!')
