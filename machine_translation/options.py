class Options(object):
    """Opções do modelo do tradutor."""

    def __init__(self, flags):
        # Opções para geração do vocabulário
        self.vocab_size = flags.vocab_size
        self.path_pt = flags.path_pt
        self.path_en = flags.path_en

        # Opções para criação do modelo
        self.seq_length = flags.seq_length
        self.batch_size = flags.batch_size
        self.embedding_dim = flags.embedding_dim
        self.memory_dim = flags.memory_dim

        # Opções para treinamento
        self.iterations = flags.iterations
        self.learning_rate = flags.learning_rate
        self.momentum = flags.momentum

        # Opções para salvar e carregar o modelo
        self.save_path = flags.save_path
        self.load_model = flags.load_model
