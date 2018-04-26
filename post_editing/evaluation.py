import argparse
from read_ape import ApeReader

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-f', '--file', help='Caminho para o arquivo anotado',
                    type=str, default=None, required=True)
PARSER.add_argument('-p', '--precision',
                    help='Precisão para a impressão das medidas', type=int, default=2)
FLAGS = PARSER.parse_args()


def MMR(ape_reader):
    mmr = 0
    for k in ape_reader.corrections:
        flat = [sub[1] for sub in k]
        try:
            reciprocal_rank = 1/(flat.index("green") + 1)
        except ValueError:
            reciprocal_rank = 0
        finally:
            mmr += reciprocal_rank
    mmr /= len(ape_reader.corrections)

    return mmr


def MAP(ape_reader):
    _map = 0
    for k in ape_reader.corrections:
        flat = [sub[1] for sub in k]
        indices = [i for (i, x) in enumerate(flat) if x == 'green']
        avep = 0
        for i in indices:
            avep += (indices.index(i) + 1) / (i + 1)
        avep /= len(flat)
        _map += avep
    _map /= len(ape_reader.corrections)
    return _map


ape = ApeReader(FLAGS.file)

print('Medida MMR: {}'.format(round(MMR(ape), FLAGS.precision)))
print('Medida MAP: {}'.format(round(MAP(ape), FLAGS.precision)))
