#!/usr/bin/env python3

import argparse
import progressbar

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i',
                    help='Caminho para o arquivo anotado pelo TreeTagger',
                    required=True,
                    type=str)
parser.add_argument('--output', '-o',
                    help='Caminho para o arquivo em que as sentenças anotadas serão salvas',
                    required=False,
                    type=str)

FLAGS = parser.parse_args()

output_path = FLAGS.output if FLAGS.output else FLAGS.input + '.sents'

sentences = ''
with open(FLAGS.input, 'r') as _file:
    lines = _file.readlines()
    for line in progressbar.progressbar(lines):
        s_line = line.split()
        try:
            if s_line[1] == 'SENT':
                sentences += s_line[0]
                sentences += '\n'
            else:
                sentences += s_line[0]
                sentences += '_'
                sentences += s_line[1]
                sentences += ' '
        except IndexError:
            sentences += s_line[0]
            sentences += ' '

with open(output_path, 'w') as _file:
    _file.write(sentences)

print('Finalizado!')