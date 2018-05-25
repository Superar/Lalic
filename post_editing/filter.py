from readers.read_ape import ApeReader

FILE_PATH = 'FAPESP_NMT_test-a_truecased_300_APE_lex-incTrWord'

ape_reader = ApeReader(FILE_PATH)

print('@annotations')
print('-1')

cores = list()
for k in ape_reader.corrections:
    flat = [sub[1] for sub in k]
    cores.append(flat)

for (i, x) in enumerate(cores):
    if 'red' in x or 'green' in x or 'yellow' in x:
        print(' '.join(ape_reader.src_lines[i]))
        print(' '.join(ape_reader.ref_lines[i]))
        print(' '.join(ape_reader.sys_lines[i]))
        error_info = [','.join(map(str, e))
                    for e in ape_reader.error_lines[i][:-1]]
        error_info.append(ape_reader.error_lines[i][-1])
        print('#'.join(error_info), end='')
        print('#@'.join(['-.-'.join(w) for w in ape_reader.corrections[i]]))
