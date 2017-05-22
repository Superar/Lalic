import argparse
from tradutor_backend import Tradutor


parser = argparse.ArgumentParser()
parser.add_argument('-pt', '--path_pt', help='Caminho para texto em português',
                    type=str, default=None)
parser.add_argument('-en', '--path_en', help='Caminho para texto em inglês',
                    type=str, default=None)
parser.add_argument('-save', '--save_path',
                    help='Caminho para diretório onde o modelo será salvo ou de onde será carregado',
                    type=str, default=None, required=True)
parser.add_argument('-vsize', '--vocabulary_size', help='Tamanho do vocabulário',
                    type=int, default=50000)
parser.add_argument('-esize', '--embedding_size', help='Dimensões para as word embeddings',
                    type=int, default=64)
parser.add_argument('-hsize', '--hidden_size', help='Dimensões para as redes neurais',
                    type=int, default=512)
parser.add_argument('-slen', '--sequence_length', help='Tamanho máximo para as sequências',
                    type=int, default=128)
parser.add_argument('-i', '--iterations', help='Número de iterações para o treinamento',
                    type=int, default=10)
parser.add_argument('-l', '--load', help='Carrega o modelo',
                    action='store_true')

FLAGS = parser.parse_args()


if FLAGS.load:
    print('Carregando modelo...')
    tradutor = Tradutor(FLAGS)
    tradutor.evaluate()
else:
    if not FLAGS.path_pt or not FLAGS.path_en:
        parser.error('The following arguments are required: -pt/--path_pt, -en/path_en')
    else:
        tradutor = Tradutor(FLAGS)
        tradutor.train()
        tradutor.save()
        tradutor.evaluate()
