import numpy as np

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