import sys
sys.path.insert(0, '/home/marciolima/Documentos/Lalic/word_embeddings')
from read_blast import BlastReader
from read_muse_embeds import read_embeds
from scipy.spatial.distance import cosine

BLAST_FILE_PATH = '/home/marciolima/Dropbox/Marcio/Tarefas/1.Anotacao_erros_corpus_NMT/Blast/Entrada_Blast/test-a/FAPESP_NMT_test-a_truecased.txt'
MUSE_EN_FILE_PATH = '/home/marciolima/MUSE/data/wiki.multi.en.vec'
MUSE_PT_FILE_PATH = '/home/marciolima/MUSE/data/wiki.multi.pt.vec'

blast_reader = BlastReader(BLAST_FILE_PATH)
errors = blast_reader.get_filtered_errors(['lex-incTrWord'])
print read_embeds(MUSE_PT_FILE_PATH)[5]
