from read_ape import ApeReader

FILE_PATH = '/home/marciolima/Downloads/FAPESP_NMT_test-a_truecased_300_APE_lex-incTrWord'

ape_reader = ApeReader(FILE_PATH)

print('Ocorrencias do erro: {}'.format(len(ape_reader.error_lines)))

cores = list()
for k in ape_reader.corrections:
    flat = [sub[1] for sub in k]
    cores.append(flat)

print('\nEfetivamente avaliadas: {}'.format(
    len([x for x in cores if 'red' in x or 'green' in x or 'yellow' in x])))

print('\nPelo menos uma sugestao correta: {}'.format(
    len([x for x in cores if 'green' in x])))

print('\nPelo menos uma sugestao parcialmente correta: {}'.format(
    len([x for x in cores if 'yellow' in x])))

print('\nPelo menos uma sugestao parcialmente correta e nenhuma correta: {}'.format(
    len([x for x in cores if 'yellow' in x and 'green' not in x])))

print('\nPelo menos uma sugestao errada: {}'.format(
    len([x for x in cores if 'red' in x])))

print('\nTodas as sugestoes erradas: {}'.format(
    len([x for x in cores if 'red' in x and 'green' not in x and 'yellow' not in x])))
