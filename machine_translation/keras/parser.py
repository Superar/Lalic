import argparse


class Arguments(object):
    """ Classe contendo os argumentos e parâmetros para o tradutor.
    Os argumentos são:

    path_pt = Caminho para texto de treinamento em português

    path_en = Caminho para o texto de treinamento em inglês

    save_path = Caminho para diretório onde o modelo será/está salvo

    vocabulary_size = Tamhanho do vocabulário

    embedding_size = Número de dimensões para as word embeddings

    hidden_size = Número de dimensões para a representação interna da sentença

    sequence_length = Tamanho máximo para as sequências a serem passadas à LSTM

    iterations = Número de epochs para o treinamento

    load = Booleano indicando se o modelo será carregado

    """

    def __init__(self):
        self._parser = argparse.ArgumentParser()

        self._parser.add_argument('-pt', '--path_pt', help='Caminho para texto em português',
                                  type=str, default=None)
        self._parser.add_argument('-en', '--path_en', help='Caminho para texto em inglês',
                                  type=str, default=None)
        self._parser.add_argument('-save', '--save_path',
                                  help='Caminho para diretório onde o modelo será salvo ou de onde será carregado',
                                  type=str, default=None, required=True)
        self._parser.add_argument('-vsize', '--vocabulary_size', help='Tamanho do vocabulário',
                                  type=int, default=50000)
        self._parser.add_argument('-esize', '--embedding_size', help='Dimensões para as word embeddings',
                                  type=int, default=64)
        self._parser.add_argument('-hsize', '--hidden_size', help='Dimensões para as redes neurais',
                                  type=int, default=512)
        self._parser.add_argument('-slen', '--sequence_length', help='Tamanho máximo para as sequências',
                                  type=int, default=128)
        self._parser.add_argument('-i', '--iterations', help='Número de iterações para o treinamento',
                                  type=int, default=10)
        self._parser.add_argument('-l', '--load', help='Carrega o modelo',
                                  action='store_true')

        self._FLAGS = self._parser.parse_args()

    def getFLAGS(self):
        return self._FLAGS
