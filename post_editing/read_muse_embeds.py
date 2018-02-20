import numpy as np
from scipy.spatial.distance import cosine

def read_embeds(filepath):
    global size, dim
    embeds = list()

    with open(filepath, 'r') as _file:
        size, dim = map(int, _file.readline().split())
        while True:
            line = _file.readline().split()
            if not line:
                break

            embed = np.array(line[-dim:], dtype=np.float)
            embeds.append((" ".join(line[:-dim]), embed))

    return embeds

def load_embeddings(path_en, path_pt):
    file_en = open(path_en, 'r')
    file_pt = open(path_pt, 'r')

    emb_en = dict()
    emb_pt = dict()

    num_emb, dim = map(int, file_en.readline().split())
    file_pt.readline()

    for _ in range(num_emb):
        line_pt = file_pt.readline().split()
        key_pt = ' '.join(line_pt[:-dim])
        value_pt = list(map(float, line_pt[-dim:]))
        emb_pt[key_pt] = np.array(value_pt)

        line_en = file_en.readline().split()
        key_en = ' '.join(line_en[:-dim])
        value_en = list(map(float, line_en[-dim:]))
        emb_en[key_en] = np.array(value_en)

    file_en.close()
    file_pt.close()
    
    return emb_en, emb_pt

def closest_words(word, emb_en, emb_pt):
    try:
        u = emb_en[word]
    except KeyError:
        return ['***']
    else:
        close = [(k, cosine(u, emb_pt[k])) for k in list(emb_pt.keys())]
        close = sorted(close, key=lambda x: x[1])
        return close[:5]