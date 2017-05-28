# import argparse
import h5py
from parser import Arguments
from tradutor_backend import Tradutor

arg = Arguments()
FLAGS = arg.getFLAGS()

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
